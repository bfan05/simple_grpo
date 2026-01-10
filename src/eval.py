import json
from typing import Dict, Any, List, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

from .rewards import extract_model_final

@torch.no_grad()
def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    out_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Deterministic eval by default (temperature=0).
    dataset must contain: prompt, answer_final (optional but used for accuracy)
    """
    model.eval()

    correct = 0
    total = 0
    avg_len = 0.0

    rows: List[Dict[str, Any]] = []

    for ex in tqdm(dataset, desc="Evaluating", leave=True):
        prompt = ex["prompt"]
        gt = ex.get("answer_final")

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

        # decode only the generated continuation
        gen_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        completion = gen_text[len(prompt_text):]

        pred = extract_model_final(completion)

        is_correct = (gt is not None and pred is not None and pred == gt)
        correct += int(is_correct)
        total += 1
        avg_len += float(len(gen[0]) - inputs["input_ids"].shape[-1])

        rows.append({
            "prompt": prompt,
            "gt": gt,
            "pred": pred,
            "completion": completion,
            "correct": bool(is_correct),
        })

    avg_len /= max(total, 1)
    acc = correct / max(total, 1)

    if out_path is not None:
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    return {
        "eval_accuracy": acc,
        "eval_avg_new_tokens": avg_len,
    }

@torch.no_grad()
def evaluate_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    out_path: Optional[str] = None,
    batch_size: int = 8,
    micro_batch_size: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()

    correct = 0
    total = 0
    avg_len = 0.0

    rows: List[Dict[str, Any]] = []

    # Ensure we have a pad token for padding batches
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    n = len(dataset)

    for start in tqdm(range(0, n, batch_size), desc="Evaluating", leave=True):
        end = min(start + batch_size, n)
        batch = dataset.select(range(start, end))  # keeps order; simple & reliable

        prompts = list(batch["prompt"])
        gts = list(batch["answer_final"]) if "answer_final" in batch.column_names else [None] * len(prompts)

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,  # set True + max_length if you need it
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        # optional micro-batching to reduce peak memory
        mb = micro_batch_size or len(prompts)
        for mb_start in range(0, len(prompts), mb):
            mb_end = min(mb_start + mb, len(prompts))

            mb_enc = {k: v[mb_start:mb_end] for k, v in enc.items()}
            mb_prompts = prompts[mb_start:mb_end]
            mb_gts = gts[mb_start:mb_end]

            gen = model.generate(
                **mb_enc,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=False,
            )

            # gen: (B, prompt_len_padded + new_tokens_padded) typically
            input_ids = mb_enc["input_ids"]
            attn = mb_enc.get("attention_mask", None)

            for i in range(gen.shape[0]):
                # true (unpadded) prompt length for this example
                if attn is not None:
                    prompt_len = int(attn[i].sum().item())
                else:
                    # fallback if no attention_mask
                    prompt_len = int((input_ids[i] != tokenizer.pad_token_id).sum().item())

                # generated continuation token ids
                continuation_ids = gen[i, prompt_len:]
                completion = tokenizer.decode(continuation_ids, skip_special_tokens=True)

                pred = extract_model_final(completion)
                gt = mb_gts[i]

                is_correct = (gt is not None and pred is not None and pred == gt)
                correct += int(is_correct)
                total += 1
                avg_len += float(continuation_ids.shape[0])

                rows.append(
                    {
                        "prompt": mb_prompts[i],
                        "gt": gt,
                        "pred": pred,
                        "completion": completion,
                        "correct": bool(is_correct),
                    }
                )

    avg_len /= max(total, 1)
    acc = correct / max(total, 1)

    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    return {"eval_accuracy": acc, "eval_avg_new_tokens": avg_len}
