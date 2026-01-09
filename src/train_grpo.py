import os
import random
from typing import Optional

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from trl import GRPOTrainer, GRPOConfig

from src.rewards import extract_gsm8k_gt, make_reward_fn
from src.eval import evaluate

from transformers import TrainerCallback
import wandb

class WandbForceHistoryCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or wandb.run is None:
            return

        wandb.log(
            {
                **logs,
                "_step": state.global_step,   # CRITICAL
            },
            commit=True,
        )

SYSTEM_PROMPT = "You are a helpful assistant. Solve the problem. Give reasoning, then the final answer."

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def make_prompt(question: str) -> str:
    return (
        f"<system>\n{SYSTEM_PROMPT}\n</system>\n"
        f"<user>\n{question}\n</user>\n"
        f"<assistant>\nLet's think step by step.\n"
    )

def load_gsm8k_train(n: Optional[int] = None) -> Dataset:
    ds = load_dataset("gsm8k", "main", split="train")
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    def _map(ex):
        return {
            "prompt": make_prompt(ex["question"]),
            "answer_final": extract_gsm8k_gt(ex["answer"]),
        }

    return ds.map(_map, remove_columns=ds.column_names)

def load_gsm8k_eval(n: int = 64) -> Dataset:
    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.select(range(min(n, len(ds))))

    def _map(ex):
        return {
            "prompt": make_prompt(ex["question"]),
            "answer_final": extract_gsm8k_gt(ex["answer"]),
        }

    return ds.map(_map, remove_columns=ds.column_names)

def main():
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    output_dir = os.environ.get("OUTPUT_DIR", "outputs/debug_grpo")
    max_steps = int(os.environ.get("MAX_STEPS", "10"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "96"))
    num_generations = int(os.environ.get("NUM_GENERATIONS", "2"))  # K
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "1"))
    lr = float(os.environ.get("LR", "5e-6"))
    seed = int(os.environ.get("SEED", "0"))
    eval_every = int(os.environ.get("EVAL_EVERY", "0"))  # 0 disables
    n_train = int(os.environ.get("N_TRAIN", "128"))
    n_eval = int(os.environ.get("N_EVAL", "64"))

    os.environ["WANDB_PROJECT"] = "simple_grpo"
    os.environ["WANDB_RUN_GROUP"] = "local_debug"
    os.environ["WANDB_NAME"] = f"grpo_{model_name.split('/')[-1]}_K{num_generations}_steps{max_steps}"

    set_seed(seed)
    device = get_device()
    print(f"[device] {device}")
    print(f"[model] {model_name}")

    # MPS fallback can help on Mac
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For CUDA, let HF place weights; for Mac/CPU just load normally.
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)

    train_ds = load_gsm8k_train(n=n_train)
    eval_ds = load_gsm8k_eval(n=n_eval)

    prompt_to_gt = {ex["prompt"]: ex["answer_final"] for ex in train_ds}
    reward_fn = make_reward_fn(prompt_to_gt)

    grpo_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=1,
        save_steps=0 if max_steps <= 50 else 50,
        max_steps=max_steps,
        seed=seed,
        # generation knobs
        generation_batch_size=num_generations,
        max_completion_length=max_new_tokens,
        num_generations=num_generations,
        # wandb
        report_to="wandb",
        run_name=os.environ["WANDB_NAME"],
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        callbacks=[WandbForceHistoryCallback()],
    )

    # run a quick eval before training
    pre = evaluate(model, tokenizer, eval_ds, max_new_tokens, temperature=0.0)
    print("[eval before]", pre)

    trainer.train()

    # Save model
    trainer.save_model(output_dir)

    # eval after training
    post = evaluate(model, tokenizer, eval_ds, max_new_tokens, temperature=0.0,
                    out_path=os.path.join(output_dir, "eval_samples.jsonl"))
    print("[eval after]", post)

if __name__ == "__main__":
    main()
