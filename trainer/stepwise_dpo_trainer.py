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
from transformers import TrainingArguments
import torch
from typing import Dict, List

class StepwiseDPOTrainer(DPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = ["chosen_scores", "rejected_scores"] 
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override the original loss to use aggregated stepwise rewards.
        inputs must contain:
        - chosen_scores: List[int]
        - rejected_scores: List[int]
        """
        # Step 1: Aggregate scores (simple mean)
        r_chosen = torch.tensor([sum(x) / len(x) for x in inputs["chosen_scores"]])
        r_rejected = torch.tensor([sum(x) / len(x) for x in inputs["rejected_scores"]])

        # Step 2: Preference loss: -log(sigmoid(r_chosen - r_rejected))
        loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()

        return (loss, None) if return_outputs else loss
