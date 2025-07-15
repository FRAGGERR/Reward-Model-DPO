import os
import time
import torch
from transformers import AutoModelForCausalLM as HFModel, AutoTokenizer
from ctransformers import AutoModelForCausalLM as GGUFModel

class TinyLlamaRewardScorer:
    def __init__(self, model_path: str):
        self.model_path = model_path

        if model_path.endswith(".gguf"):
            print(f"Loading GGUF model from: {model_path}")
            start = time.time()
            self.model = GGUFModel.from_pretrained(
                model_path,
                model_type="llama",
                gpu_layers=0  # Use CPU
            )
            self.tokenizer = None  # Not needed
            self.is_gguf = True
            print(f"GGUF model loaded in {time.time() - start:.2f}s.")

        else:
            print(f"Loading Hugging Face model from: {model_path}")
            start = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = HFModel.from_pretrained(model_path, local_files_only=True)
            self.model.eval()
            self.is_gguf = False
            print(f"HF model loaded in {time.time() - start:.2f}s.")

    def score_step(self, prompt: str, step: str, step_id: int = None) -> int:
        if step_id is not None:
            print(f"\n[Step {step_id}] Scoring...")

        input_text = (
            f"{prompt}\n\nIs the following reasoning step correct?\n{step}\n\nAnswer yes or no:"
        )

        if self.is_gguf:
            print("Using GGUF model...")
            response = self.model(input_text, max_new_tokens=5).lower()

        else:
            print("Using Hugging Face model...")
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=5
                )
            response = self.tokenizer.decode(output[0], skip_special_tokens=True).lower()

        print(f"Model response: {response}")
        score = 1 if "yes" in response else 0
        print(f"Step score: {score}")
        return score
