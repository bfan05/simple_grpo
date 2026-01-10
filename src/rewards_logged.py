import hashlib

from typing import Dict, Optional, List, Callable
from .rewards import (
    extract_model_final,
    normalize_num,
    make_reward_fn,
)
from .rollout_logger import RolloutRecorder

def prompt_hash(prompt: str, n: int = 12) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:n]

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

            records.append({
                "step": recorder.step,
                "prompt_id": prompt_hash(p),
                "prompt": p,
                "completion": c,
                "prediction": pred,
                "ground_truth": gt,
                "reward": float(r),
                "correct": (pred is not None and gt is not None and pred == gt),
            })

        recorder.write(records)
        return rewards

    return reward_fn
