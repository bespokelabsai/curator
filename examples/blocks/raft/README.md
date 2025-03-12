# RAFT: Retrieval Augmented Fine-Tuning

This example folder provides an implementation of RAFT (Retrieval Augmented Fine-Tuning) for adapting Large Language Models (LLMs) to domain-specific Retrieval-Augmented Generation (RAG) using Curator.

## Installation

First, install the required dependencies:

```shell
pip install -r requirements.txt
```

## Setup

### Export Environment Variables
Before running the scripts, configure your environment variables:

```shell
# Prepare the environment
export PDF_URL="https://arxiv.org/pdf/2503.03323.pdf"
export OCR_BACKEND="aryn"  # Options: "aryn" or "local" (pdfplumber)
export DISTRIBUTED=1  # Enable multi-GPU training
export OPENAI_API_KEY=<your_openai_api_key>
```

## OCR Backend Configuration
If backend is set to `aryn`, you need to set aryn api key:
goto [Aryn](https://console.aryn.ai/api-keys) and get your api key and set it as an environment variable:

```shell
export ARYN_API_KEY=<you_key>
```

## Dataset Preparation

Prepare the dataset using RAFT with **Curator**:

```shell
PYTHONPATH=./:.. python3 examples/blocks/raft/raft.py
```

This step processes domain-specific documents, extracts text, generates questions, and prepares data for fine-tuning the LLM.

## Fine-Tune Llama-3.1-8B-Instruct Model

### Single GPU Training
Run fine-tuning on a single GPU:

```shell
python3 train.py
```

### Multi-GPU Training with DeepSpeed
For distributed training across multiple GPUs, use DeepSpeed:

```shell
DISTRIBUTED=1 deepspeed --num_gpus=4 examples/blocks/raft/train.py
```

## Running RAFT for Inference
After fine-tuning, perform inference using the RAFT fine-tuned model:

```shell
PYTHONPATH=./:.. python3 examples/blocks/raft/run.py
```
