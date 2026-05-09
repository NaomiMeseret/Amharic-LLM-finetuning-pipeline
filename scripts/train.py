import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# ── 1. config ────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "./outputs/amharic-llama"
MAX_SEQ_LENGTH = 512

# ── 2. load tokenizer ────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── 3. load dataset ──────────────────────────────────────────
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/processed/train.jsonl",
        "validation": "data/processed/val.jsonl"
    }
)

def format_prompt(example):
    text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    return {"text": text}

dataset = dataset.map(format_prompt)
print(f"✓ Train: {len(dataset['train']):,} examples")
print(f"✓ Validation: {len(dataset['validation']):,} examples")

# ── 4. quantization config ───────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ── 5. load model ────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)

# ── 6. LoRA config ───────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 7. training arguments ────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    eval_steps=200,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    eval_strategy="steps",
    report_to="none",
    save_total_limit=2,
)

# ── 8. trainer ───────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

# ── 9. train ─────────────────────────────────────────────────
print("Starting training...")
trainer.train()

# ── 10. save ─────────────────────────────────────────────────
model.save_pretrained(f"{OUTPUT_DIR}/adapters")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapters")
print(f"✓ Adapters saved to {OUTPUT_DIR}/adapters")