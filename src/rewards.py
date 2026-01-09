import re
from typing import Optional, Dict, Any, List, Callable

# GSM8K ground-truth format uses: "#### <final answer>"
FINAL_ANSWER_RE = re.compile(r"####\s*([^\n]+)")

def extract_gsm8k_gt(answer_field: str) -> Optional[str]:
    """Extract the canonical GSM8K final answer from the dataset 'answer' field."""
    m = FINAL_ANSWER_RE.search(answer_field)
    return m.group(1).strip() if m else None

def extract_model_final(text: str) -> Optional[str]:
    """
    Minimal extraction for model output:
    - grab the last number-like token (handles commas)
    """
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1].strip() if nums else None

def compute_reward_components(gt: Optional[str], completion: str) -> Dict[str, float]:
    """
    Returns individual reward components so you can log/debug.
    Keep components small/bounded; combine later.
    """
    pred = extract_model_final(completion)

    exact = 1.0 if (gt is not None and pred is not None and pred == gt) else 0.0
    has_digit = 1.0 if re.search(r"\d", completion) else 0.0

    # tiny shaping bonus to encourage producing an answer-like output
    shaping = 0.05 * has_digit

    return {
        "exact": exact,
        "shaping": shaping,
        "pred_none": 1.0 if pred is None else 0.0,
    }

def combine_components(components: Dict[str, float]) -> float:
    r = components["exact"] + components["shaping"]
    # clamp to [0, 1]
    return float(max(0.0, min(1.0, r)))

def make_reward_fn(prompt_to_gt: Dict[str, Optional[str]]) -> Callable[[List[str], List[str]], List[float]]:
    """
    Returns a TRL-compatible reward function for GRPOTrainer in your TRL version:
      reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]
    """
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards: List[float] = []
        for p, c in zip(prompts, completions):
            gt = prompt_to_gt.get(p)
            pred = extract_model_final(c)

            r = 1.0 if (gt is not None and pred is not None and pred == gt) else 0.0
            if re.search(r"\d", c):
                r += 0.05
            rewards.append(float(max(0.0, min(1.0, r))))
        return rewards

    return reward_fn