# Amharic LLM Fine-Tuning

Fine-tuning a Large Language Model for Amharic — 
a low-resource Semitic language spoken by 37 million 
people in Ethiopia — using QLoRA on LLaMA 3.2 1B.

## Overview

Amharic is severely underrepresented in most large 
language model training corpora, accounting for less 
than 0.01% of CommonCrawl data. This project 
demonstrates a practical, low-budget approach to 
adapting an open-source LLM for Amharic 
English-to-Amharic translation using parameter- 
efficient fine-tuning.

## Key Findings

- LLaMA's default tokenizer fragments Amharic words 
  into an average of 11 raw byte tokens vs 2 tokens 
  for equivalent English words — a 6.5x inflation ratio
- QLoRA reduces VRAM requirement from ~56GB (full 
  fine-tuning) to under 4GB for a 1B parameter model
- 61,064 clean parallel Amharic-English pairs were 
  prepared from the OPUS Bible corpus after preprocessing
- Only 0.27% of model parameters are trained using 
  LoRA adapters (3.4M out of 1.2B)

## Model

- Base model: meta-llama/Llama-3.2-1B
- Method: QLoRA (4-bit NF4 quantization + LoRA rank 16)
- Task: English to Amharic translation
- Hardware: Google Colab T4 GPU (free tier)

## Dataset

Primary: OPUS Bible Corpus (61,084 parallel pairs)
https://object.pouta.csc.fi/OPUS-bible-uedin/v1/moses/am-en.txt.zip

Additional sources evaluated:
- CC-100 Amharic: data.statmt.org/cc-100
- FLORES-200: huggingface.co/datasets/facebook/flores
- MasakhaNER: huggingface.co/datasets/masakhane/masakhaner2

## Quick Start

# 1. Clone the repository
git clone https://github.com/yourusername/amharic-llm-finetune
cd amharic-llm-finetune

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download and preprocess data
python scripts/preprocess.py

# 4. Train the model
python scripts/train.py

# 5. Run inference
python scripts/inference.py

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | LLaMA 3.2 1B |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Quantization | 4-bit NF4 |
| Batch Size | 2 (effective 16) |
| Learning Rate | 2e-4 |
| Epochs | 1 (prototype) |
| GPU | T4 16GB |

## Cost Estimate

| Resource | Cost |
|----------|------|
| Data | Free (open source) |
| Prototype training (Colab T4) | Free |
| Full run (RunPod A100) | ~$15-40 |
| Total | Under $50 |

## Limitations

- Training data is limited to religious domain (Bible text)
  and does not represent modern conversational Amharic
- LLaMA's tokenizer was not trained on Ge'ez script,
  causing significant token fragmentation
- Prototype run used 5,000 examples due to free-tier
  GPU constraints
- Formal BLEU evaluation pending full training run

## References

- QLoRA paper: arxiv.org/abs/2305.14314
- LoRA paper: arxiv.org/abs/2106.09685
- FLORES-200: arxiv.org/abs/2207.04672
- MasakhaNER 2.0: aclanthology.org/2022.emnlp-main.298
- Masakhane community: masakhane.io
