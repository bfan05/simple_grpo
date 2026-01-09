import re
from typing import Optional, Dict, List, Callable

# GSM8K ground-truth format uses: "#### <final answer>"
FINAL_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
FINAL_RE = re.compile(r"Final:\s*(-?\d+(?:\.\d+)?)")
FINAL_AT_END_RE = re.compile(
    r"""
    ^[\s\S]*?          # anything before (non-greedy)
    \n?                # optional newline
    Final:\s*          # literal 'Final:'
    (-?\d+(?:\.\d+)?)  # the number
    \s*                # optional trailing whitespace
    $                  # END OF STRING (nothing after)
    """,
    re.VERBOSE,
)

def extract_gsm8k_gt(answer_field: str) -> Optional[str]:
    """Extract the canonical GSM8K final answer from the dataset 'answer' field."""
    m = FINAL_ANSWER_RE.search(answer_field)
    return m.group(1).strip() if m else None

def extract_model_final(text: str) -> Optional[str]:
    """
    Extracts the final answer ONLY if it is the last line and
    strictly matches: Final: <number>
    """
    text = text.replace(",", "").strip()
    m = FINAL_AT_END_RE.match(text)
    return m.group(1) if m else None

def normalize_num(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    try:
        return str(float(s)) if "." in s else str(int(s))
    except ValueError:
        return None

def make_reward_fn(prompt_to_gt: Dict[str, Optional[str]]) -> Callable:
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards: List[float] = []

        for p, c in zip(prompts, completions):
            gt = normalize_num(prompt_to_gt.get(p))
            pred = normalize_num(extract_model_final(c))

            if pred is None:
                # missing Final OR text after Final
                r = 0.0
            elif gt is not None and pred == gt:
                # correct format AND correct answer
                r = 1.0
            else:
                # correct format, wrong answer
                r = 0.1

            rewards.append(float(r))

        return rewards

    return reward_fn