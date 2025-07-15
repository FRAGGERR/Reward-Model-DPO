# train.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from trainer.stepwise_dpo_trainer import StepwiseDPOTrainer
from utils.load_rewarded_dataset import load_rewarded_dataset
from trl import DPOConfig, DPOTrainer
import torch
import os
def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # change if you're using another

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    # Load dataset
    dataset = load_rewarded_dataset()

    # Tokenize prompt + responses
    def tokenize(sample):
        prompt_input_ids = tokenizer(
            sample["prompt"],
            truncation=True,
            padding="max_length",
            max_length=128
        ).input_ids

        chosen_input_ids = tokenizer(
            sample["chosen"],
            truncation=True,
            padding="max_length",
            max_length=128
        ).input_ids

        rejected_input_ids = tokenizer(
            sample["rejected"],
            truncation=True,
            padding="max_length",
            max_length=128
        ).input_ids

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "chosen_scores": sample["chosen_scores"],      # ✅ returned explicitly
            "rejected_scores": sample["rejected_scores"]   # ✅ returned explicitly
        }



    dataset = dataset.map(tokenize)

    # Set training args
    output_dir = "saved_model/stepwise_dpo_tinyllama"
    os.makedirs(output_dir, exist_ok=True)

    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=1,
        save_total_limit=1,
        save_strategy="no",
        remove_unused_columns=False,
        fp16=False,
        bf16=False,
        disable_tqdm=False,
        report_to=None
    )

    # Initialize trainer
    trainer = StepwiseDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    # Start training
    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()