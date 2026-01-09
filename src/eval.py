import json
from typing import Dict, Any, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

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

    for ex in dataset:
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
        completion = gen_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)) :]

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

    return {"eval_accuracy": acc, "eval_avg_new_tokens": avg_len}
