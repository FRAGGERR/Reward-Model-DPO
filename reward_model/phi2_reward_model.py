from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Phi2RewardScorer:
    def __init__(self, model_name: str = "microsoft/phi-2"):
        print("Loading Phi-2 model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        self.model.eval()

        # if torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        # else:
        #     self.device = torch.device("cpu")

        self.device = torch.device("cpu")

        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def score_step(self, prompt: str, step: str) -> int:
        """
        Returns 1 if the step seems correct based on the prompt; otherwise 0.
        For now, uses log-likelihood hack.
        """

        full_input = f"{prompt}\n\nIs the following reasoning step correct?\n{step}\n\nAnswer yes or no:"
        inputs = self.tokenizer(full_input, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=5)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

        if "yes" in response:
            return 1
        elif "no" in response:
            return 0
        else:
            return 0  # Default fallback