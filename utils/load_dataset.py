import random
from typing import List, Dict

def load_synthetic_stepwise_dataset(sample_size: int = 3) -> List[Dict]:
    examples = []

    for i in range(sample_size):
        prompt = "What is 17 * 12? Show your steps."

        correct = (
            "Step 1: Break 17 into 10 and 7.\n"
            "Step 2: 10 * 12 = 120\n"
            "Step 3: 7 * 12 = 84\n"
            "Step 4: Add: 120 + 84 = 204\n"
            "Answer: 204"
        )

        incorrect = (
            "Step 1: Break 17 into 10 and 7.\n"
            "Step 2: 10 * 12 = 120\n"
            "Step 3: 7 * 12 = 64\n"  # wrong step
            "Step 4: Add: 120 + 64 = 184\n"
            "Answer: 184"
        )

        # Randomize chosen/rejected order
        if random.random() < 0.5:
            chosen, rejected = correct, incorrect
        else:
            chosen, rejected = incorrect, correct

        examples.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

    print(f"Loaded {len(examples)} synthetic examples.")
    return examples