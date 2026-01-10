## simple_grpo

**simple_grpo** is a minimal, end-to-end reinforcement learning (RL) training stack for LLM **reasoning** built on top of Hugging Face TRL’s `GRPOTrainer`. It is intentionally small and opinionated, with the goal of being used to understand the full RL pipeline rather than serving as a production system.

The current setup focuses on **GSM8K** math word problems and walks through:

- **Rollouts**: sampling multiple completions per prompt
- **Rewards**: turning GSM8K answers into scalar rewards
- **GRPO updates**: policy optimization via TRL’s `GRPOTrainer`
- **Evaluation**: greedy GSM8K accuracy before/after RL
- **Logging**: lightweight Weights & Biases (W&B) integration

The code is structured so you can quickly modify rewards, datasets, models, and training schedules to run small experiments.

---

## Repository structure

- **`src/train_grpo.py`**: Main training script
  - Loads model/tokenizer
  - Builds GSM8K train/eval datasets
  - Constructs the GRPO trainer and reward function
  - Supports YAML config files via `--config` flag or environment variables
  - Saves model checkpoints and rollout logs
- **`src/rewards.py`**
  - Utilities for parsing GSM8K answers
  - Extracts ground truth from GSM8K format (`#### <answer>`)
  - Extracts model predictions from strict `Final: <number>` format
  - Reward function: 1.0 for correct, 0.1 for wrong but correct format, 0.0 for missing format
- **`src/rewards_logged.py`**
  - Wraps the base reward function to log all rollouts to JSONL
  - Records prompt, completion, prediction, ground truth, reward, and correctness
- **`src/rollout_logger.py`**
  - `RolloutRecorder`: Writes rollout records to `rollouts.jsonl`
  - `StepTrackerCallback`: Keeps rollout step counter in sync with trainer's global_step
- **`src/eval.py`**
  - Deterministic (or sampled) evaluation loop on GSM8K
  - Computes accuracy and average generation length
  - Optionally logs per-example results to JSONL
- **`configs/local_debug.yaml`**
  - Example config for Mac / CPU / MPS debugging
  - Includes model, dataset, training, and runtime settings
- **`configs/aws_config.yaml`**
  - Example config targeting a GPU box (e.g., AWS)
  - Similar structure to `local_debug.yaml` with GPU-optimized settings
- **`scripts/setup_local.sh`**
  - Convenience script to create a conda env and install dependencies on macOS (Apple Silicon) with CPU/MPS PyTorch
- **`scripts/run_train.sh`**
  - Wrapper around `python -m src.train_grpo --config <path>`
  - Defaults to `configs/local_debug.yaml` if no argument provided
- **`scripts/plot_prompt_accuracy.py`**
  - Utility to plot accuracy over training steps for a specific prompt
  - Reads from `rollouts.jsonl` and generates accuracy vs step plots

---

## High-level training flow

The **training loop** in `src/train_grpo.py` works as follows:

1. **Configuration**
   - Supports YAML config files via `--config` flag (e.g., `configs/local_debug.yaml`)
   - Environment variables override YAML values (e.g., `MODEL_NAME`, `OUTPUT_DIR`, `MAX_STEPS`)
   - Defaults: `Qwen/Qwen2.5-0.5B-Instruct`, `outputs/debug_grpo`, `MAX_STEPS=10`, etc.
   - Config structure: `model_name`, `max_new_tokens`, `num_generations`, `train.*`, `dataset.*`, `runtime.*`
2. **Device + seeding**
   - Uses CUDA if available, otherwise MPS on Apple Silicon, otherwise CPU
   - On MPS, enables `PYTORCH_ENABLE_MPS_FALLBACK=1` by default (configurable via `runtime.mps_fallback`)
3. **Model & tokenizer**
   - Loads a causal LM (`AutoModelForCausalLM`) and tokenizer (`AutoTokenizer`)
   - Sets `pad_token` to `eos_token` if missing
   - On CUDA, uses `torch_dtype=torch.bfloat16` and `device_map="auto"`
4. **Data & prompts**
   - Loads **GSM8K** train/test splits via `datasets`
   - For each example:
     - Builds a chat-style prompt using the tokenizer's chat template:
       - System: "You are a helpful assistant. Solve the problem.\nShow your reasoning briefly.\nAt the end, output exactly one line in this format:\nFinal: <number>\nDo not write anything after the Final line."
       - User: GSM8K question
     - Extracts the canonical GSM8K final answer from the `#### <answer>` format
5. **Reward function & logging**
   - Constructs a dict `prompt_to_gt: prompt -> final_answer`
   - Creates a `RolloutRecorder` that writes to `{output_dir}/rollouts/rollouts.jsonl`
   - Wraps the base reward function with `make_logged_reward_fn` to log all rollouts:
     - Each rollout record includes: step, prompt_id (hash), prompt, completion, prediction, ground_truth, reward, correct
   - The reward function signature: `reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]`
6. **GRPO configuration**
   - Creates `GRPOConfig` with:
     - `output_dir`, `learning_rate`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `max_steps`, `seed`
     - **Generation-specific settings**:
       - `generation_batch_size = batch_size * num_generations`
       - `max_completion_length = max_new_tokens`
       - `num_generations = num_generations` (K rollouts per prompt)
     - Logging:
       - `logging_steps` (from config, default 1)
       - `save_steps` (from config, default 0)
       - `report_to="wandb"` if `runtime.use_wandb` is true, else `"none"`
       - `run_name` auto-generated from model name, K, and steps
7. **Trainer + callbacks**
   - Instantiates `GRPOTrainer` with:
     - `model`, `args=grpo_args`, `train_dataset=train_ds`
     - `processing_class=tokenizer`
     - `reward_funcs=reward_fn` (the logged reward function)
     - `callbacks=[StepTrackerCallback(recorder)]`
   - `StepTrackerCallback` keeps `recorder.step` in sync with `trainer.global_step` for accurate rollout logging
8. **Training & saving**
   - Runs `trainer.train()`
   - Saves the final model and tokenizer to `output_dir`
   - Note: Evaluation calls are currently commented out in the code

This gives a compact **end-to-end RL pipeline**: prompts → rollouts → rewards → GRPO updates → saved model + rollout logs.

---

## Environment setup

### macOS (Apple Silicon) – local debug

The most direct way to get started on a Mac is to use the provided setup script:

1. **Install Miniforge/conda** (if you don’t already have it).
2. From the project root:

```bash
chmod +x scripts/setup_local.sh
./scripts/setup_local.sh
```

What this does:
- Creates/uses a conda env named `rl-local` with Python 3.10
- Installs CPU/MPS versions of:
  - `torch`, `transformers`, `datasets`, `accelerate`, `trl`, `peft`
  - `sentencepiece`, `wandb`, `evaluate`, `numpy`, `scipy`, `packaging`, `rich`, `matplotlib`
- Runs a small sanity check to print Torch version and CUDA/MPS availability

Alternatively, if you already have an environment:

```bash
pip install -r requirements.txt
```

Make sure your PyTorch install is compatible with your hardware (CPU/MPS, not CUDA wheels on Mac).

### GPU (e.g., AWS with NVIDIA GPUs)

On an AWS GPU instance (e.g., G5), the workflow is:

1. **Create a Python environment** (conda or venv).
2. Install **CUDA-enabled PyTorch** following the official PyTorch instructions for your CUDA version.
3. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

4. Optionally set some environment variables for stability:

```bash
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

5. Use `configs/aws_config.yaml` as a reference for reasonable hyperparameters on GPU (see "Running a debug training job" below).

---

## Running a debug training job

The intended first step is a **small, fast GRPO run** on GSM8K to verify everything is wired correctly.

### Minimal local debug run (using config file)

From the project root, after activating your environment:

```bash
python -m src.train_grpo --config configs/local_debug.yaml
```

Or use the convenience script:

```bash
./scripts/run_train.sh configs/local_debug.yaml
```

The config file specifies:
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Training steps, batch size, learning rate, etc.
- Dataset sizes (`n_train`, `n_eval`)
- Runtime settings (W&B, MPS fallback)

### Using environment variables

You can override config values or run without a config file using environment variables:

```bash
export MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
export OUTPUT_DIR=outputs/my_experiment
export MAX_STEPS=50
export NUM_GENERATIONS=4
export MAX_NEW_TOKENS=128
export BATCH_SIZE=1
export GRAD_ACCUM=2
export LR=2e-6
export N_TRAIN=256
export N_EVAL=128

python -m src.train_grpo
```

The script will:
- Print device/model info
- Run GRPO training
- Save the final model and tokenizer to `OUTPUT_DIR`
- Write rollout logs to `{OUTPUT_DIR}/rollouts/rollouts.jsonl`

Note: Evaluation calls are currently commented out in the code, but you can uncomment them in `train_grpo.py` if needed.

---

## GSM8K reward design

All reward logic lives in `src/rewards.py`.

- **Ground truth extraction**
  - GSM8K’s official `answer` field often contains a worked solution ending with a line like:
    - `#### 42`
  - `extract_gsm8k_gt(answer_field)` uses a regex to pull out this final number.
  - `train_grpo.py` stores this as `answer_final`.

- **Model answer extraction**
  - The system prompt instructs the model to output: `Final: <number>` as the last line
  - `extract_model_final(text)` uses a strict regex that:
    - Only matches if `Final: <number>` appears at the very end of the completion
    - Strips commas and whitespace
    - Returns the number if the format is correct, `None` otherwise
  - This enforces a structured output format rather than extracting arbitrary numbers.

- **Current reward function (used by GRPO)**
  - `make_reward_fn(prompt_to_gt)` builds a function that:
    - For each `(prompt, completion)`:
      - Looks up the ground-truth final answer from `prompt_to_gt[prompt]`
      - Extracts the model's final answer using `extract_model_final(completion)`
      - Normalizes both numbers (converts to float/int string representation)
      - Reward logic:
        - `1.0` if `pred == gt` (correct format AND correct answer)
        - `0.1` if format is correct but answer is wrong
        - `0.0` if `Final: <number>` format is missing or malformed

  - Result: a **format-enforcing reward** that strongly penalizes missing the required output structure.

- **Reward logging**
  - `make_logged_reward_fn` (in `src/rewards_logged.py`) wraps the base reward function
  - Logs every rollout to `{output_dir}/rollouts/rollouts.jsonl` with:
    - Step number, prompt hash, full prompt/completion, prediction, ground truth, reward, correctness flag
  - This enables post-hoc analysis of how rewards evolve during training

This design is intentionally simple so you can **modify it freely**:
- Add partial credit (e.g., close numeric distance)
- Penalize very long or malformed completions
- Reward intermediate reasoning structure (e.g., “The answer is …” lines)

---

## Evaluation setup

Evaluation logic is implemented in `src/eval.py`:

- **API**
  - `evaluate(model, tokenizer, dataset, max_new_tokens, temperature=0.0, top_p=1.0, out_path=None) -> Dict[str, float]`

- **Dataset expectations**
  - Each example should contain:
    - `prompt`: the full input prompt (system + user, formatted via chat template)
    - `answer_final` (optional, but needed for accuracy)

- **Generation**
  - Calls `model.generate` with:
    - `max_new_tokens=max_new_tokens`
    - Greedy decoding by default (`temperature=0.0`)
    - Optional sampling if `temperature > 0`
  - Decodes only the **newly generated part** as the completion.

- **Metrics**
  - Extracts the final numeric answer from the completion using `extract_model_final` (same logic as rewards).
  - Computes:
    - `eval_accuracy`: fraction of examples where `pred == gt`
    - `eval_avg_new_tokens`: average number of generated tokens

- **Outputs**
  - If `out_path` is provided, writes one JSON line per example with:
    - `prompt`, `gt`, `pred`, `completion`, `correct`

**Note**: In `train_grpo.py`, the evaluation calls are currently commented out. You can uncomment them to run evaluation before and after training if needed.

---

## Weights & Biases logging

Logging is designed to be minimal but sufficient for small experiments.

- **Configuration**
  - W&B is controlled via the config file: `runtime.use_wandb` (default: `false` in code, `true` in example configs)
  - If enabled, the script sets:
    - `WANDB_PROJECT = "simple_grpo"` (can be overridden via environment variable)
  - The run name is auto-generated: `grpo_{model_name}_K{num_generations}_steps{max_steps}`
  - You can override these via environment variables **before** launching:

    ```bash
    export WANDB_PROJECT=my-rl-project
    export WANDB_RUN_GROUP=exp1
    export WANDB_NAME=grpo_qwen2_0p5b_debug
    python -m src.train_grpo --config configs/local_debug.yaml
    ```

- **Integration with TRL**
  - `GRPOConfig(report_to="wandb" if use_wandb else "none")` enables/disables W&B logging.
  - Standard Trainer / GRPO metrics are logged automatically (losses, learning rate, etc.)

- **Rollout logging**
  - All rollouts are logged to `{output_dir}/rollouts/rollouts.jsonl` regardless of W&B settings
  - Use `scripts/plot_prompt_accuracy.py` to visualize accuracy over time for specific prompts:
    ```bash
    python scripts/plot_prompt_accuracy.py \
      --rollouts outputs/local_debug/rollouts/rollouts.jsonl \
      --prompt_id <hash> \
      --num_generations 8 \
      --out outputs/local_debug/rollouts/plots/<prompt_id>.png
    ```

To use W&B:
- Make sure `wandb` is installed.
- Run `wandb login` once on the machine.
- Set `runtime.use_wandb: true` in your config file.

---

If you’re using this repo, you are encouraged to **treat it as a sandbox**:
copy the reward function, plug in your own dataset, and iterate quickly to build intuition about what actually improves reasoning under RL. 

