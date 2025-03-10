# RAFT: Retrieval Augmented Fine-Tuning

This example folder provides an implementation of RAFT (Retrieval Augmented Fine-Tuning) for adapting Large Language Models (LLMs) to domain-specific Retrieval-Augmented Generation (RAG) using Curator.

## Installation

First, install the required dependencies:

```shell
pip install -r reqs.txt
```

## Setup

### Export Environment Variables
Before running the scripts, configure your environment variables:

```shell
# Prepare the environment
export ARXIV_ID="2503.03323"
export OCR_BACKEND="aryn"  # Options: "aryn" or "python" (pdfplumber)
export DISTRIBUTED=1  # Enable multi-GPU training
```

## Dataset Preparation

Prepare the dataset using RAFT with **Curator**:

```shell
python3 raft.py
```

This step processes domain-specific documents, extracts text, generates questions, and prepares data for fine-tuning the LLM.

## Fine-Tune Llama 3 (8B) Model

### Single GPU Training
Run fine-tuning on a single GPU:

```shell
python3 train.py
```

### Multi-GPU Training with DeepSpeed
For distributed training across multiple GPUs, use DeepSpeed:

```shell
DISTRIBUTED=1 deepspeed --num_gpus=4 train.py
```

## Running RAFT for Inference
After fine-tuning, perform inference using the RAFT fine-tuned model:

```shell
python3 run.py
```
