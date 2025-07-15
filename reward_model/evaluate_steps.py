import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from tqdm import tqdm
import os
from reward_model.tinyllama_reward_model import TinyLlamaRewardScorer

INPUT_PATH = "../data/synthetic_dataset.jsonl"
OUTPUT_PATH = "../data/rewarded.jsonl"

def score_steps(scorer, prompt, steps):
    scores = []
    for step in steps:
        try:
            score = scorer.score_step(prompt, step)
        except Exception as e:
            print(f"Error scoring step: {step}\n{e}")
            score = 0  # fallback score
        scores.append(score)
    return scores

def evaluate_dataset():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Dataset not found at {INPUT_PATH}")

    scorer = TinyLlamaRewardScorer()
    results = []

    with open(INPUT_PATH, "r") as f:
        lines = f.readlines()

    print(f"Scoring {len(lines)} examples...")

    for line in tqdm(lines):
        item = json.loads(line)
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_scores = score_steps(scorer, prompt, chosen)
        rejected_scores = score_steps(scorer, prompt, rejected)

        result = {
            "prompt": prompt,
            "chosen": chosen,
            "chosen_scores": chosen_scores,
            "rejected": rejected,
            "rejected_scores": rejected_scores
        }
        results.append(result)

    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            json.dump(r, f)
            f.write("\n")

    print(f"Saved scored dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    evaluate_dataset()