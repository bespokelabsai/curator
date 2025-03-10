## Installation
```shell
pip install -r reqs.txt
```

## Setup
### Export environment variables
```shell
# Prepare the environment
export ARXIV_ID="2001.00001"
export MODEL_PATH="llama3-finetuned/final"
export DISTRIBUTED=1
```
### Run dataset preparation

```shell
python3 raft.py
```

### Finetune Llama3 (8B) model

#### Run single GPU training
```shell
python3 train.py
```
#### Run single GPU training
```shell
DISTRIBUTED=1 deepspeed --num_gpus=4 train.py
```

### Perform raft
```shell
python3 run.py
```
