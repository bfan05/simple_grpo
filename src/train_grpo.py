import os
import random
from typing import Optional

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from trl import GRPOTrainer, GRPOConfig

from src.rewards import extract_gsm8k_gt, make_reward_fn
from src.eval import evaluate, evaluate_batched
from src.rollout_logger import RolloutRecorder, StepTrackerCallback
from src.rewards_logged import make_logged_reward_fn

import argparse
import yaml

def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}

SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the problem.\n"
    "Show your reasoning briefly.\n"
    "At the end, output exactly one line in this format:\n"
    "Final: <number>\n"
    "Do not write anything after the Final line."
)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def make_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def load_gsm8k_train(tokenizer, n: Optional[int] = None) -> Dataset:
    ds = load_dataset("gsm8k", "main", split="train")
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    def _map(ex):
        return {
            "prompt": make_prompt(tokenizer, ex["question"]),
            "answer_final": extract_gsm8k_gt(ex["answer"]),
        }

    return ds.map(_map, remove_columns=ds.column_names)

def load_gsm8k_eval(tokenizer, n: int = 64) -> Dataset:
    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.select(range(min(n, len(ds))))

    def _map(ex):
        return {
            "prompt": make_prompt(tokenizer, ex["question"]),
            "answer_final": extract_gsm8k_gt(ex["answer"]),
        }

    return ds.map(_map, remove_columns=ds.column_names)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config) if args.config else {}
    train_cfg = cfg.get("train", {}) or {}
    dataset_cfg = cfg.get("dataset", {}) or {}
    runtime_cfg = cfg.get("runtime", {}) or {}

    model_name = os.environ.get(
        "MODEL_NAME", cfg.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    )
    output_dir = os.environ.get(
        "OUTPUT_DIR", train_cfg.get("output_dir", "outputs/debug_grpo")
    )

    max_steps = int(os.environ.get("MAX_STEPS", train_cfg.get("num_train_steps", 10)))
    max_new_tokens = int(
        os.environ.get("MAX_NEW_TOKENS", cfg.get("max_new_tokens", 96))
    )
    num_generations = int(
        os.environ.get("NUM_GENERATIONS", cfg.get("num_generations", 2))
    )

    batch_size = int(
        os.environ.get("BATCH_SIZE", train_cfg.get("per_device_train_batch_size", 1))
    )
    grad_accum = int(
        os.environ.get("GRAD_ACCUM", train_cfg.get("gradient_accumulation_steps", 1))
    )
    lr = float(os.environ.get("LR", train_cfg.get("learning_rate", 5e-6)))
    seed = int(os.environ.get("SEED", train_cfg.get("seed", 0)))

    logging_steps = int(train_cfg.get("logging_steps", 1))
    save_steps = int(train_cfg.get("save_steps", 0))

    n_train = int(os.environ.get("N_TRAIN", dataset_cfg.get("n_train", 128)))
    n_eval = int(os.environ.get("N_EVAL", dataset_cfg.get("n_eval", 64)))
    eval_batch_size = int(
        os.environ.get("EVAL_BATCH_SIZE", dataset_cfg.get("eval_batch_size", 16))
    )
    eval_micro_batch_size = int(
        os.environ.get(
            "EVAL_MICRO_BATCH_SIZE", dataset_cfg.get("eval_micro_batch_size", 4)
        )
    )

    # Runtime toggles from YAML
    use_wandb = bool(runtime_cfg.get("use_wandb", False))
    mps_fallback = bool(runtime_cfg.get("mps_fallback", True))

    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", "simple_grpo")
    else:
        os.environ["WANDB_MODE"] = "disabled"

    set_seed(seed)
    device = get_device()
    print(f"[device] {device}")
    print(f"[model] {model_name}")

    if device == "mps" and mps_fallback:
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

    train_ds = load_gsm8k_train(tokenizer=tokenizer, n=n_train)
    eval_ds = load_gsm8k_eval(tokenizer=tokenizer, n=n_eval)

    prompt_to_gt = {ex["prompt"]: ex["answer_final"] for ex in train_ds}

    rollout_dir = os.path.join(output_dir, "rollouts")
    recorder = RolloutRecorder(out_dir=rollout_dir)

    reward_fn = make_logged_reward_fn(prompt_to_gt, recorder)

    grpo_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=logging_steps,
        save_steps=save_steps,
        max_steps=max_steps,
        seed=seed,
        # generation knobs
        generation_batch_size=batch_size * num_generations,
        max_completion_length=max_new_tokens,
        num_generations=num_generations,
        # wandb
        report_to="wandb" if use_wandb else "none",
        run_name=f"grpo_{model_name.split('/')[-1]}_K{num_generations}_steps{max_steps}",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    trainer.add_callback(StepTrackerCallback(recorder))

    # run a quick eval before training
    pre = evaluate_batched(
        model,
        tokenizer,
        eval_ds,
        max_new_tokens,
        temperature=0.0,
        out_path=os.path.join(output_dir, "eval_samples_before.jsonl"),
        batch_size=eval_batch_size,
        micro_batch_size=eval_micro_batch_size,
    )
    print("[eval before]", pre)

    trainer.train()

    # Save model
    trainer.save_model(output_dir)

    # eval after training
    post = evaluate_batched(
        model,
        tokenizer,
        eval_ds,
        max_new_tokens,
        temperature=0.0,
        out_path=os.path.join(output_dir, "eval_samples_after.jsonl"),
        batch_size=eval_batch_size,
        micro_batch_size=eval_micro_batch_size,
    )
    print("[eval after]", post)

if __name__ == "__main__":
    main()
