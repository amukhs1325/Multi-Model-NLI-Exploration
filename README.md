# Multi-Model NLI Exploration

Comparing fine-tuned transformers and prompted LLMs on multilingual Natural Language Inference.

Given a **premise** and a **hypothesis**, the task is to decide whether the hypothesis is a **contradiction** (0), **neutral** (1), or **entailment** (2). The data is the Kaggle [Contradictory, My Dear Watson](https://www.kaggle.com/c/contradictory-my-dear-watson/data) set: about 12,000 sentence pairs in 15 languages, roughly 56% English.

## What we tried

| Approach | Where | Notes |
|---|---|---|
| Fine-tuned mBERT (`bert-base-multilingual-cased`) | notebook, `train_transformer.py` | Baseline with early stopping, plus a two-phase run (train the head first, then the full model) |
| Fine-tuned RoBERTa-base | notebook | Layer-wise learning-rate decay and early stopping |
| Fine-tuned XLM-R base / large, pre-trained on XNLI | notebook, `train_transformer.py` | `morit/xlm-t-roberta-base-mnli-xnli`, `joeddav/xlm-roberta-large-xnli` |
| Fine-tuned GPT-2 | notebook | Decoder-only model with a classification head |
| GPT-4 prompting | notebook, `predict_with_gpt4.py` | Few-shot in the notebook, zero-shot in the script |
| Local LLMs with Ollama | `predict_with_ollama.py`, `ensemble.py` | Few-shot prompts with multilingual examples, self-consistency (5 samples from Gemma 3 12B), and a majority vote with Llama 3 8B |

## Results

Validation set: 1,212 held-out pairs (stratified 10% split, `random_state=42`).

| Model | Accuracy | Macro-F1 |
|---|---|---|
| XLM-RoBERTa-large (XNLI) fine-tuned | **0.92** | **0.92** |
| XLM-RoBERTa-base (XNLI) fine-tuned | 0.80 | 0.80 |
| GPT-4 few-shot (75-row sample) | 0.87 | 0.86 |
| mBERT, two-phase fine-tuning | 0.68 | 0.68 |
| mBERT fine-tuned | 0.65 | 0.65 |
| RoBERTa-base fine-tuned | 0.65 | 0.65 |
| GPT-2 fine-tuned | 0.60 | 0.60 |

The biggest gain came from starting with a model that had already seen multilingual NLI (XNLI). English-only or general multilingual checkpoints topped out in the 0.60s, while the XNLI checkpoints reached 0.80 to 0.92. The GPT-4 number is on a small sample, so treat it as a rough comparison.

## Project layout

```
Multi‐Model_NLI_Exploration.ipynb   EDA, label check, and all fine-tuning experiments
split_train_val.py                  Flips labels to 0=contra/1=neutral/2=entail and writes val.csv
sample.py                           Prints examples per label (how we spotted the flipped labels)
train_transformer.py                Command-line fine-tuning for any Hugging Face checkpoint
predict_with_gpt4.py                OpenAI zero-shot predictions on a random sample
predict_with_ollama.py              Few-shot + self-consistency with a local model
ensemble.py                         Gemma self-consistency + Llama vote
evaluate.py                         Scores val_preds.csv
compare_models.py                   Scores many prediction files side by side, overall and by language
```

A note on labels: in the original Kaggle file, 0 means entailment and 2 means contradiction. Every script here flips them so that 0 = contradiction, 1 = neutral, 2 = entailment.

## Running it

```bash
pip install -r requirements.txt

# Fine-tune a transformer (writes preds_<model>.csv)
python train_transformer.py --model bert-base-multilingual-cased
python train_transformer.py --model joeddav/xlm-roberta-large-xnli --epochs 3 --lr 1e-5 --batch-size 8

# Prompt an OpenAI model
export OPENAI_API_KEY=...
python predict_with_gpt4.py --n 200 --model gpt-4o-mini

# Prompt local models (needs Ollama running, with gemma3:12b and llama3:8b pulled)
python predict_with_ollama.py
python ensemble.py

# Compare everything
python compare_models.py --by-language
```

A GPU is strongly recommended for the fine-tuning runs. If XLM-R large runs out of memory, lower `--batch-size`.

## Team

Built by [Tejas Mutneja](https://github.com/TejasMutneja) and [Agastya Mukherjee](https://github.com/amukhs1325). Originally developed in [TejasMutneja/Multi-Model-NLI-Exploration](https://github.com/TejasMutneja/Multi-Model-NLI-Exploration).
