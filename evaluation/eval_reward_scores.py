import os
import json
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from reward_model.tinyllama_reward_model import TinyLlamaRewardScorer

# Paths (relative)
REWARDED_PATH = "/Users/hardikchhipa/Desktop/Data_Manipulations_Projects/futureagi-ml-intern/data/rewarded.jsonl"
OUTPUT_PATH = "evaluation/reward_score_comparison.json"
BASE_MODEL_PATH = "/Users/hardikchhipa/Desktop/Data_Manipulations_Projects/futureagi-ml-intern/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
TUNED_MODEL_PATH = "/Users/hardikchhipa/Desktop/Data_Manipulations_Projects/futureagi-ml-intern/saved_model/stepwise_dpo_tinyllama"  # Not used with ctransformers

def average_score(score_list):
    return sum(score_list) / len(score_list) if score_list else 0

def evaluate_reward_scores():
    print("Loading base model...")
    base_scorer = TinyLlamaRewardScorer(BASE_MODEL_PATH)

    print("Loading fine-tuned model...")
    tuned_scorer = TinyLlamaRewardScorer(TUNED_MODEL_PATH)

    with open(REWARDED_PATH, "r") as f:
        lines = f.readlines()

    results = []

    print(f"Scoring {len(lines)} examples...")

    for line in tqdm(lines):
        item = json.loads(line)
        prompt = item["prompt"]

        # Join steps into single string for each variant
        chosen_text = "\n".join(item["chosen"])
        rejected_text = "\n".join(item["rejected"])

        # Score with base model
        base_chosen_score = base_scorer.score_step(prompt, chosen_text)
        base_rejected_score = base_scorer.score_step(prompt, rejected_text)

        # Score with fine-tuned model
        tuned_chosen_score = tuned_scorer.score_step(prompt, chosen_text)
        tuned_rejected_score = tuned_scorer.score_step(prompt, rejected_text)

        results.append({
            "prompt": prompt,
            "base_score_diff": base_chosen_score - base_rejected_score,
            "tuned_score_diff": tuned_chosen_score - tuned_rejected_score
        })

    # Calculate average improvement
    base_avg = sum(r["base_score_diff"] for r in results) / len(results)
    tuned_avg = sum(r["tuned_score_diff"] for r in results) / len(results)
    improvement = tuned_avg - base_avg

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "base_avg_score_diff": base_avg,
            "tuned_avg_score_diff": tuned_avg,
            "improvement": improvement,
            "num_examples": len(results)
        }, f, indent=2)

    print(f"Saved reward comparison to {OUTPUT_PATH}")
    print(f"\nBase avg:   {base_avg:.4f}")
    print(f"Tuned avg:  {tuned_avg:.4f}")
    print(f"Improvement: {improvement:.4f}")

if __name__ == "__main__":
    evaluate_reward_scores()