import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from reward_model.phi2_reward_model import Phi2RewardScorer

# if __name__ == "__main__":
#     # Example usage
#     scorer = Phi2RewardScorer()
#     prompt = "What is 12 * 15? Show your steps."
#     step = "Step 2: Multiply 10 * 12 = 120"
#     # This is a placeholder step; replace with actual reasoning steps as needed.
#     score = scorer.score_step(prompt, step)
#     print(f"Score for the step: {score}")


from reward_model.tinyllama_reward_model import TinyLlamaRewardScorer

if __name__ == "__main__":
    scorer = TinyLlamaRewardScorer()

    prompt = "What is 17 * 12? Show your steps."
    step = "Step 2: 10 * 12 = 120"

    score = scorer.score_step(prompt, step)
    print(f"Step score: {score}")