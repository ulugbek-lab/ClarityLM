# ClarityLM


**ClarityLM** is an experimental large language model fine-tuning project built using TinyLlama-1.1B-Chat, Hugging Face Transformers, and Parameter-Efficient Fine-Tuning (PEFT).

The project explores how pretrained language models can be adapted for instruction-following tasks using **QLoRA**, combining 4-bit quantization with Low-Rank Adaptation (LoRA) to reduce GPU memory requirements.

ClarityLM is developed and experimented with in Google Colab using PyTorch.

## Project Overview

The primary goal of ClarityLM is to understand and implement the fundamental stages of LLM development, from tokenization and model architecture to efficient fine-tuning and text generation.

The project covers:

- Loading and exploring pretrained causal language models
- Tokenization, token IDs, embeddings, and Transformer architecture
- 4-bit model quantization using BitsAndBytes
- Parameter-efficient fine-tuning using LoRA
- Supervised fine-tuning with instruction-response datasets
- Model inference and generation parameter configuration
- Comparing pretrained and fine-tuned model behavior

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Pretrained model loading and text generation |
| Hugging Face Datasets | Dataset loading and preprocessing |
| PEFT / LoRA | Parameter-efficient fine-tuning |
| TRL / SFTTrainer | Supervised fine-tuning |
| BitsAndBytes | 4-bit model quantization |
| Google Colab | Notebook development and GPU experimentation |
| LangDetect | Input language detection |

## Model Architecture

**Base Model:** [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

ClarityLM builds upon TinyLlama, a compact decoder-only Transformer language model with approximately 1.1 billion parameters.

The language-model pipeline follows:

```text
User Input
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Embedding Layer
    ↓
Transformer Decoder Layers
    ↓
Language Modeling Head
    ↓
Next-Token Prediction
    ↓
Generated Response
```

The pretrained model is loaded with 4-bit quantization, reducing the memory required to store supported model weights.

## Fine-Tuning Approach

ClarityLM uses **QLoRA (Quantized Low-Rank Adaptation)**.

Instead of updating all parameters of TinyLlama, LoRA introduces small trainable matrices into selected attention projection layers while keeping the quantized base model frozen.

**LoRA configuration:**

| Parameter | Value |
|---|---|
| Rank (`r`) | 8 |
| LoRA Alpha | 16 |
| Target Modules | `q_proj`, `v_proj` |
| LoRA Dropout | 0.05 |
| Task Type | Causal Language Modeling |
| Quantization | 4-bit NF4 |
| Double Quantization | Enabled |

This approach reduces the number of trainable parameters and makes experimentation more practical on limited GPU hardware.

## Dataset

**Dataset:** [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)

The notebook uses 7,000 selected examples from the Alpaca Cleaned instruction-following dataset.

Each example contains an instruction, an optional input, and an expected response.

The data is formatted into instruction-response text for supervised fine-tuning.

## Training Configuration

| Setting | Value |
|---|---|
| Training Framework | TRL SFTTrainer |
| Per-device Batch Size | 1 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 4 on one device |
| Training Epochs | 2 |
| Checkpoint Strategy | Per epoch |
| Logging Interval | 30 steps |
| Training Method | LoRA / QLoRA |

Training uses next-token prediction loss and backpropagation to update the LoRA adapter parameters.

## Inference and Model Testing

The notebook includes experiments for generating responses using the pretrained model and loading the fine-tuned LoRA adapter.

A custom `ask_clarityLM()` function provides:

- Prompt preparation using an instruction-response format
- English-language input checking with LangDetect
- Text generation with configurable maximum output length
- Temperature and top-p sampling controls
- Response decoding and formatting

The project is experimental. Model responses may be inaccurate, and no production-level reliability guarantees are made.

## Repository Structure

```text
ClarityLM/
├── README.md
└── ClarityLM.ipynb
```

`ClarityLM.ipynb` contains the model-loading, dataset preparation, fine-tuning, saving, and inference experiments.

Trained adapter weights and dataset caches are not included in the repository.

## Getting Started

1. Open `ClarityLM.ipynb` in Google Colab.
2. Select a compatible GPU runtime.
3. Install the Python dependencies specified in the notebook.
4. Load the TinyLlama base model and tokenizer.
5. Prepare the Alpaca instruction dataset.
6. Configure quantization and LoRA.
7. Execute the supervised fine-tuning process.
8. Save the resulting adapter and evaluate generated responses.

Some training settings may require adjustment depending on the available GPU and installed library versions.

## Project Status

**Experimental / Under Development**

ClarityLM is an ongoing learning and research project focused on understanding LLM fine-tuning, Transformer architecture, and efficient model adaptation.

Potential future improvements include more systematic evaluation, improved dataset preparation, reusable inference scripts, and a conversational interface.

## Acknowledgments

This project uses the open-source TinyLlama model, the Alpaca Cleaned dataset, and the Hugging Face machine learning ecosystem.
