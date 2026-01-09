## simple_grpo

**simple_grpo** is a minimal, end-to-end reinforcement learning (RL) training stack for LLM **reasoning** built on top of Hugging Face TRL’s `GRPOTrainer`. It is intentionally small and opinionated, aimed at helping you understand the full RL pipeline rather than serving as a production system.

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
  - Runs pre/post evaluation and saves outputs
- **`src/rewards.py`**
  - Utilities for parsing GSM8K answers
  - Reward shaping for GSM8K-style numeric answers
- **`src/eval.py`**
  - Deterministic (or sampled) evaluation loop on GSM8K
  - Computes accuracy and average generation length
  - Optionally logs per-example results to JSONL
- **`configs/local_debug.yaml`**
  - Example “tiny” config for Mac / CPU / MPS debugging
  - Mirrors defaults hard-coded in `train_grpo.py`
- **`configs/box.yaml`**
  - Example higher-budget config targeting a GPU box (e.g., AWS)
- **`scripts/setup_local.sh`**
  - Convenience script to create a conda env and install dependencies on macOS (Apple Silicon) with CPU/MPS PyTorch
- **`scripts/run_train.sh`**
  - Simple wrapper around `python -m src.train_grpo` (config flag is currently unused; environment variables control behavior)
- **`outputs/`**
  - Example outputs from a debug GRPO run (`outputs/debug_grpo/`)
  - Includes final model checkpoint, tokenizer artifacts, and evaluation samples

---

## High-level training flow

The **training loop** in `src/train_grpo.py` works as follows:

1. **Configuration via environment variables**
   - `MODEL_NAME` (default: `Qwen/Qwen2.5-0.5B-Instruct`)
   - `OUTPUT_DIR` (default: `outputs/debug_grpo`)
   - `MAX_STEPS`, `MAX_NEW_TOKENS`, `NUM_GENERATIONS` (K), `BATCH_SIZE`, `GRAD_ACCUM`, `LR`
   - `N_TRAIN`, `N_EVAL`, `EVAL_EVERY` (currently unused, eval is pre and post only)
   - `SEED`
2. **Device + seeding**
   - Uses CUDA if available, otherwise MPS on Apple Silicon, otherwise CPU
   - On MPS, enables `PYTORCH_ENABLE_MPS_FALLBACK=1` to avoid unsupported ops crashes
3. **Model & tokenizer**
   - Loads a causal LM (`AutoModelForCausalLM`) and tokenizer (`AutoTokenizer`)
   - Sets `pad_token` to `eos_token` if missing
   - On CUDA, uses `torch_dtype=torch.bfloat16` and `device_map="auto"`
4. **Data & prompts**
   - Loads **GSM8K** train/test splits via `datasets`
   - For each example:
     - Builds a chat-style prompt:
       - System: “You are a helpful assistant. Solve the problem. Give reasoning, then the final answer.”
       - User: GSM8K question
       - Assistant: fixed prefix “Let’s think step by step.”
     - Extracts the canonical GSM8K final answer (see reward section)
5. **Reward function**
   - Constructs a dict `prompt_to_gt: prompt -> final_answer`
   - Wraps this into a TRL-compatible reward function:
     - Signature: `reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]`
     - Computes a scalar reward per completion (details below)
6. **GRPO configuration**
   - Creates `GRPOConfig` with:
     - `output_dir`, `learning_rate`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `max_steps`, `seed`
     - **Generation-specific settings**:
       - `generation_batch_size = NUM_GENERATIONS`
       - `max_completion_length = MAX_NEW_TOKENS`
       - `num_generations = NUM_GENERATIONS` (K rollouts per prompt)
     - Logging:
       - `logging_steps = 1`
       - `save_steps = 0 if max_steps <= 50 else 50`
       - `report_to="wandb"` and `run_name` from environment
7. **Trainer + callbacks**
   - Instantiates `GRPOTrainer` with:
     - `model`, `args=grpo_args`, `train_dataset=train_ds`
     - `processing_class=tokenizer`
     - `reward_funcs=reward_fn`
     - `callbacks=[WandbForceHistoryCallback()]`
   - `WandbForceHistoryCallback` ensures W&B logs use `_step = trainer.global_step` for clean histories.
8. **Pre/post evaluation**
   - Runs `evaluate` **before** RL training on a held-out GSM8K eval subset
   - Runs `trainer.train()`
   - Saves the final model to `OUTPUT_DIR`
   - Runs `evaluate` **after** training and optionally writes `eval_samples.jsonl`

This gives a compact **end-to-end RL pipeline**: prompts → rollouts → rewards → GRPO updates → eval → saved model + logs.

---

## Environment setup

### macOS (Apple Silicon) – local debug

The most direct way to get started on a Mac is to use the provided setup script:

1. **Install Miniforge/conda** (if you don’t already have it).
2. From the project root:

```bash
cd /Users/bfan/Projects/simple_grpo  # adjust as needed
chmod +x scripts/setup_local.sh
./scripts/setup_local.sh
```

What this does:
- Creates/uses a conda env named `rl-local` with Python 3.10
- Installs CPU/MPS versions of:
  - `torch`, `transformers`, `datasets`, `accelerate`, `trl`, `peft`
  - `sentencepiece`, `wandb`, `evaluate`, `numpy`, `scipy`, `packaging`, `rich`
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

5. Use `configs/box.yaml` as a reference for reasonable hyperparameters on GPU (see “Running a debug training job” below).

---

## Running a debug training job

The intended first step is a **small, fast GRPO run** on GSM8K to verify everything is wired correctly.

### Minimal local debug run (defaults)

From the project root, after activating your environment:

```bash
python -m src.train_grpo
```

By default this will:
- Use `Qwen/Qwen2.5-0.5B-Instruct`
- Train for `MAX_STEPS=10`
- Use `NUM_GENERATIONS=2` rollouts per prompt
- Generate up to `MAX_NEW_TOKENS=96` per completion
- Use `OUTPUT_DIR=outputs/debug_grpo`

You can customize via environment variables, e.g.:

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
- Run an **eval-before-training** on a subset of GSM8K test
- Run GRPO training
- Run an **eval-after-training**
- Save the final model and tokenizer to `OUTPUT_DIR`

### Using example configs

`configs/local_debug.yaml` and `configs/box.yaml` are provided as **reference configs**. The current training script primarily reads configuration from **environment variables**, so the YAML files are best thought of as documented presets you can manually mirror via `export` lines (or extend the script to parse `--config`).

---

## GSM8K reward design

All reward logic lives in `src/rewards.py`.

- **Ground truth extraction**
  - GSM8K’s official `answer` field often contains a worked solution ending with a line like:
    - `#### 42`
  - `extract_gsm8k_gt(answer_field)` uses a regex to pull out this final number.
  - `train_grpo.py` stores this as `answer_final`.

- **Model answer extraction**
  - `extract_model_final(text)`:
    - Strips commas, finds all number-like tokens (`-?\d+\.?\d*`)
    - Returns the **last** one as the model’s predicted answer.

- **Current reward function (used by GRPO)**
  - `make_reward_fn(prompt_to_gt)` builds a function:

    - For each `(prompt, completion)`:
      - Looks up the ground-truth final answer from `prompt_to_gt[prompt]`
      - Extracts the model’s final numeric answer from `completion`
      - If `pred == gt`:
        - Base reward = `1.0`
      - If the completion contains **any digit** at all:
        - Adds a small shaping bonus `+0.05`
      - Clamps reward to \([0, 1]\)

  - Result: a **sparse-but-binary** correctness reward with a tiny shaping term encouraging number-like outputs.

- **Alternative component API**
  - `compute_reward_components` / `combine_components` are included as a more modular interface:
    - Components: `"exact"`, `"shaping"`, `"pred_none"` etc.
    - `combine_components` currently just adds `"exact" + "shaping"` and clamps.
  - The GRPO trainer currently uses the simpler `make_reward_fn` path, but these helpers are intended to make it easy to experiment with more nuanced rewards.

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
    - `prompt`: the full input prompt (system + user + assistant prefix)
    - `answer_final` (optional, but needed for accuracy)

- **Generation**
  - Calls `model.generate` with:
    - `max_new_tokens=max_new_tokens`
    - Greedy decoding by default (`temperature=0.0`)
    - Optional sampling if `temperature > 0`
  - Decodes only the **newly generated part** as the completion.

- **Metrics**
  - Extracts the final numeric answer from the completion (same logic as rewards).
  - Computes:
    - `eval_accuracy`: fraction of examples where `pred == gt`
    - `eval_avg_new_tokens`: average number of generated tokens

- **Outputs**
  - If `out_path` is provided, writes one JSON line per example with:
    - `prompt`, `gt`, `pred`, `completion`, `correct`

`train_grpo.py` calls `evaluate`:
- Once **before** training (sanity check baseline)
- Once **after** training (with `out_path=OUTPUT_DIR/eval_samples.jsonl`), enabling deeper inspection of improvements.

---

## Weights & Biases logging

Logging is designed to be minimal but sufficient for small experiments.

- **Environment configuration**
  - In `train_grpo.py`, the script sets:

    - `WANDB_PROJECT = "simple_grpo"`
    - `WANDB_RUN_GROUP = "local_debug"`
    - `WANDB_NAME = f"grpo_{model_name}_K{num_generations}_steps{max_steps}"`

  - You can override these via environment variables **before** launching:

    ```bash
    export WANDB_PROJECT=my-rl-project
    export WANDB_RUN_GROUP=exp1
    export WANDB_NAME=grpo_qwen2_0p5b_debug
    python -m src.train_grpo
    ```

- **Integration with TRL**
  - `GRPOConfig(report_to="wandb")` enables W&B logging.
  - A custom callback `WandbForceHistoryCallback`:
    - Intercepts `on_log` events
    - Re-logs with `_step = trainer_state.global_step`
    - Ensures step-aligned plots in the W&B UI (no weird gaps or off-by-one indices).

- **What you can expect to see**
  - Standard Trainer / GRPO metrics (losses, learning rate, etc.)
  - Global step-based x-axis
  - You can add your own custom logging (e.g., average reward, accuracy on a small validation subset) via additional callbacks.

To use W&B:
- Make sure `wandb` is installed.
- Run `wandb login` once on the machine.

---

If you’re using this repo, you are encouraged to **treat it as a sandbox**:
copy the reward function, plug in your own dataset, and iterate quickly to build intuition about what actually improves reasoning under RL. 

