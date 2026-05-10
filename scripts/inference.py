# inference.py
# Run translation inference using the fine-tuned Amharic model

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_model(base_model_path, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer

def translate(model, tokenizer, english_text):
    prompt = f"""### Instruction:
Translate the following English text to Amharic.

### Input:
{english_text}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=False
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()

if __name__ == "__main__":
    model, tokenizer = load_model(
        base_model_path="meta-llama/Llama-3.2-1B",
        adapter_path="./outputs/amharic-llama/adapters"
    )
    print("Amharic Translation Model Ready")
    print("Type 'quit' to exit\n")

    while True:
        text = input("Enter English text: ")
        if text.lower() == "quit":
            break
        result = translate(model, tokenizer, text)
        print(f"Amharic: {result}\n")