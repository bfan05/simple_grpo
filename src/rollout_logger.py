import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import TrainerCallback

@dataclass
class RolloutRecorder:
    out_dir: str
    step: int = 0

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.path = os.path.join(self.out_dir, "rollouts.jsonl")

    def write(self, records: List[Dict[str, Any]]):
        if not records:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

class StepTrackerCallback(TrainerCallback):
    """
    Keeps recorder.step in sync with HF Trainer's optimizer step.
    """
    def __init__(self, recorder: RolloutRecorder):
        self.recorder = recorder

    def on_step_begin(self, args, state, control, **kwargs):
        self.recorder.step = int(state.global_step)
