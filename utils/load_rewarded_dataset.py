import json
from datasets import Dataset
from typing import List, Dict

def load_rewarded_dataset(path: str = "data/rewarded.jsonl") -> Dataset:
    """
    Load the reward-scored dataset and prepare for DPOTrainer.
    """

    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)

            data.append({
                "prompt": item["prompt"],
                "chosen": "\n".join(item["chosen"]),
                "rejected": "\n".join(item["rejected"]),
                "chosen_scores": item["chosen_scores"],
                "rejected_scores": item["rejected_scores"]
            })

    return Dataset.from_list(data)