"""
------------------------------------------------------------------------------
Stepwise DPO Trainer
------------------------------------------------------------------------------
This is where we:
  • Subclass HuggingFace's `DPOTrainer`
  • Modify `compute_loss()` to:
      → Use your `chosen_scores` and `rejected_scores`
      → Aggregate them (e.g., mean of step scores)
      → Apply preference loss:

            loss = -log(σ(R_chosen - R_rejected))
------------------------------------------------------------------------------
"""


from trl import DPOTrainer
import torch
from transformers import DataCollatorWithPadding
from torch.utils.data.dataloader import DataLoader, default_collate
from typing import Any, Dict, List

class CustomDataCollator:
    """Robust collator that converts all values to torch.Tensor as needed."""
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        for key in features[0].keys():
            first_elem = features[0][key]

            # Handle lists (e.g., chosen_scores, rejected_scores)
            if isinstance(first_elem, list):
                batch[key] = torch.tensor([f[key] for f in features], dtype=torch.float32)
            # Handle tensors
            elif isinstance(first_elem, torch.Tensor):
                batch[key] = torch.stack([f[key] for f in features])
            # Handle strings (for compatibility with trainer expectations)
            elif isinstance(first_elem, str):
                batch[key] = [f[key] for f in features]
            # Fail loudly on unknown types
            else:
                raise TypeError(f"Unsupported type for key '{key}': {type(first_elem)}")
        return batch


class StepwiseDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_collator = CustomDataCollator()
        self.label_names = ["chosen_scores", "rejected_scores"]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Verify scores exist
        if "chosen_scores" not in inputs or "rejected_scores" not in inputs:
            missing = [k for k in ["chosen_scores", "rejected_scores"] if k not in inputs]
            raise ValueError(f"Missing score fields in inputs: {missing}")

        device = model.device

        # Convert list of step scores to tensors (no real backprop — Option A)
        chosen_scores = torch.tensor(
            [sum(scores)/len(scores) for scores in inputs["chosen_scores"]],
            dtype=torch.float32,
            device=device
        )
        rejected_scores = torch.tensor(
            [sum(scores)/len(scores) for scores in inputs["rejected_scores"]],
            dtype=torch.float32,
            device=device
        )

        # Preference loss: -log(sigmoid(R_c - R_r))
        diff = chosen_scores - rejected_scores
        loss = -torch.nn.functional.logsigmoid(diff).mean()

        return (loss, None) if return_outputs else loss

    def training_step(self, model, inputs, num_items=None):
        model.eval()

        loss = self.compute_loss(model, inputs)

        if isinstance(loss, tuple):
            loss = loss[0]

        print(f"Step loss: {loss.item():.4f}")
        return loss