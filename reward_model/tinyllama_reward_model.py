from ctransformers import AutoModelForCausalLM
import os
import time

class TinyLlamaRewardScorer:
    def __init__(self, model_path="../models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
        print("Loading TinyLlama 4-bit model...")
        start = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="llama",
            gpu_layers=0,  # Force CPU mode
        )
        print(f"Model loaded (CPU 4-bit) in {time.time() - start:.2f}s.")

    def score_step(self, prompt: str, step: str, step_id: int = None) -> int:
        if step_id is not None:
            print(f"\n [Step {step_id}] Scoring step...")

        input_text = (
            f"{prompt}\n\nIs the following reasoning step correct?\n{step}\n\nAnswer yes or no:"
        )

        print(f"Prompting model...")
        response = self.model(input_text, max_new_tokens=5).lower()
        print(f"Model response: {response}")

        score = 1 if "yes" in response else 0
        print(f"Step score: {score}")
        return score