import hashlib
import re

from typing import Dict, Optional, List, Callable
from .rewards import (
    extract_model_final,
    normalize_num,
    make_reward_fn,
)
from .rollout_logger import RolloutRecorder

try:
    import rollout_viz
except ImportError:
    rollout_viz = None

def prompt_hash(prompt: str, n: int = 12) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:n]

def extract_question_from_prompt(prompt: str) -> str:
    """
    Extract the user question from a formatted prompt.
    Works with Qwen-style chat templates and other common formats.
    """
    # Pattern: <|im_start|>user\n...question...<|im_end|>
    match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try to find user content in other formats
    match = re.search(r'user[^>]*>\s*(.*?)(?:<|assistant|system)', prompt, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""  # Return empty if we can't extract

def make_logged_reward_fn(
    prompt_to_gt: Dict[str, Optional[str]],
    recorder: RolloutRecorder,
) -> Callable:
    """
    Wraps your existing reward fn and logs every rollout.
    """

    base_reward_fn = make_reward_fn(prompt_to_gt)

    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = base_reward_fn(prompts, completions, **kwargs)

        records = []
        for p, c, r in zip(prompts, completions, rewards):
            gt = normalize_num(prompt_to_gt.get(p))
            pred = normalize_num(extract_model_final(c))
            
            # Extract question text from prompt for consistent tracking
            question = extract_question_from_prompt(p)
            # Use question hash as question_id if we extracted a question, otherwise use prompt_id
            question_id = prompt_hash(question) if question else prompt_hash(p)

            record = {
                "step": recorder.step,
                "prompt_id": prompt_hash(p),
                "question_id": question_id,
                "question": question,
                "prompt": p,
                "completion": c,
                "prediction": pred,
                "ground_truth": gt,
                "reward": float(r),
                "correct": (pred is not None and gt is not None and pred == gt),
            }
            records.append(record)
            if rollout_viz is not None:
                rollout_viz.log_rollout(record)
        if rollout_viz is None:
            recorder.write(records)
        return rewards

    return reward_fn
