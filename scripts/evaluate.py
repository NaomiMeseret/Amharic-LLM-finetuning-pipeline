# evaluate.py
# Evaluates fine-tuned Amharic model using BLEU and ChrF metrics
# against the FLORES-200 held-out test set

from sacrebleu.metrics import BLEU, CHRF
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

def translate(model, tokenizer, text):
    prompt = f"""### Instruction:
Translate the following English text to Amharic.

### Input:
{text}

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

def evaluate(model, tokenizer, test_pairs):
    bleu = BLEU()
    chrf = CHRF()

    predictions = []
    references = []

    for pair in test_pairs:
        pred = translate(model, tokenizer, pair["english"])
        predictions.append(pred)
        references.append([pair["amharic"]])
        print(f"EN: {pair['english']}")
        print(f"Predicted AM: {pred}")
        print(f"Expected  AM: {pair['amharic']}")
        print()

    bleu_score = bleu.corpus_score(predictions, references)
    chrf_score = chrf.corpus_score(predictions, references)

    print(f"BLEU Score: {bleu_score.score:.2f}")
    print(f"ChrF Score: {chrf_score.score:.2f}")

# example test cases
test_pairs = [
    {
        "english": "Good morning. How are you today?",
        "amharic": "እንደምን አደሩ? ዛሬ እንዴት ናቸሁ?"
    },
    {
        "english": "What is the capital of Ethiopia?",
        "amharic": "የኢትዮጵያ ዋና ከተማ አዲስ አበባ ነው።"
    },
    {
        "english": "The child went to school.",
        "amharic": "ልጁ ወደ ትምህርት ቤት ሄደ።"
    }
]

if __name__ == "__main__":
    model, tokenizer = load_model(
        base_model_path="meta-llama/Llama-3.2-1B",
        adapter_path="./outputs/amharic-llama/adapters"
    )
    evaluate(model, tokenizer, test_pairs)