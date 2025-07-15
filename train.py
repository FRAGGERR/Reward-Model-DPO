# train.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from trainer.stepwise_dpo_trainer import StepwiseDPOTrainer
from utils.load_rewarded_dataset import load_rewarded_dataset
from trl import DPOConfig
import torch
import os

def tokenize_function(sample, tokenizer):
    prompt = tokenizer(
        sample["prompt"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    chosen = tokenizer(
        sample["chosen"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    rejected = tokenizer(
        sample["rejected"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    return {
        "input_ids_prompt": prompt["input_ids"][0],
        "attention_mask_prompt": prompt["attention_mask"][0],
        "input_ids_chosen": chosen["input_ids"][0],
        "attention_mask_chosen": chosen["attention_mask"][0],
        "input_ids_rejected": rejected["input_ids"][0],
        "attention_mask_rejected": rejected["attention_mask"][0],
        "chosen_scores": sample["chosen_scores"],
        "rejected_scores": sample["rejected_scores"],
        "chosen": sample["chosen"],
        "rejected": sample["rejected"]
    }

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = load_rewarded_dataset()
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=dataset.column_names
    )
    print(f"Tokenized dataset features: {tokenized_dataset.features}")

    # Create output directory
    output_dir = "saved_model/stepwise_dpo_tinyllama"
    os.makedirs(output_dir, exist_ok=True)

    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1, #2
        num_train_epochs=1, #3
        logging_dir="./logs",
        logging_steps=1,
        save_total_limit=1,
        save_strategy="no",
        remove_unused_columns=False,
        fp16=False,
        bf16=False,
        disable_tqdm=False,
        report_to=None,
        optim="adamw_torch"
    )

    trainer = StepwiseDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        # preprocess_ref_log_probs=False,
        data_collator=None, 
        # preprocess_dataset=False
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()