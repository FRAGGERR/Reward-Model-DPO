import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_default_device("cpu")

def format_prompt(user_input: str) -> str:
    """
    Wrap input in <|system|><|user|><|assistant|> tags.
    """
    return f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{user_input}\n<|assistant|>\n"

def generate_response(prompt: str, model, tokenizer, max_new_tokens=100): #if it takes too long, decrease this number to 20.
    """
    Generate a model response given a prompt.
    """
    formatted = format_prompt(prompt)

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        print("Generating response...")
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("<|assistant|>")[-1].strip()

def main():
    model_path = "saved_model/stepwise_dpo_tinyllama"  # Path to your fine-tuned model

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    model.eval()

    print("Inference ready. Type your prompt:")
    while True:
        user_input = input(">> ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = generate_response(user_input, model, tokenizer)
        print("\nAssistant:\n" + response + "\n")

if __name__ == "__main__":
    main()