import json
import matplotlib.pyplot as plt
import os

# Load reward score results
with open("evaluation/evaluation_results/reward_score_comparison.json", "r") as f:
    results = json.load(f)

# Extract values
base = results["base_avg_score_diff"]
tuned = results["tuned_avg_score_diff"]
improvement = results["improvement"]

# Plot
plt.figure(figsize=(6, 4))
bars = plt.bar(["Base Model", "Tuned Model"], [base, tuned], color=["gray", "green"])
plt.title("Average Reward Score Comparison")
plt.ylabel("Average Step Score")
plt.ylim(0, max(base, tuned, 0.1) + 0.05)

# Add values on top
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.2f}", ha='center', va='bottom')

# Save
os.makedirs("evaluation/plots", exist_ok=True)
plt.tight_layout()
plt.savefig("evaluation/plots/reward_score_comparison.png")
print("✅ Saved plot to evaluation/plots/reward_score_comparison.png")
plt.show()