# Using [Bespoke Curator](https://github.com/bespokelabsai/curator) to generate the [Bespoke-Stratos](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) dataset.

## Setup

Recommended using python 3.12

```bash
pip install -r requirements.txt
```

## Generate reasoning traces from DeepSeek-R1 using Curator

Our final dataset [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) contains the following subsets:

* Numina: 10.5k samples from the `math`, `olympiads`, and `amc_aime` subset of the [difficulty-labeled Numina dataset](https://huggingface.co/datasets/NovaSky-AI/labeled_numina_difficulty_162K).
* APPS: ~2.5k samples from the [APPS dataset](https://huggingface.co/datasets/codeparrot/apps).
* TACO: ~3k samples from the [TACO dataset](https://huggingface.co/datasets/BAAI/TACO).
* STILL-2: ~1k samples from the [STILL-2 dataset](https://huggingface.co/datasets/RUC-AIBOX/long_form_thought_data_5k).

Run the following scripts to generate reasoning traces and filter out incorrect reasoning traces:

```bash
# Numina Math
python generate_numina_data.py

# APPS
python generate_apps_data.py --split test

# TACO
python generate_taco_data.py --split train
python generate_taco_data.py --split test
```

Run the following to combine datasets:

```bash
python combine_data.py
```

## Training and Evaluation

Using the dataset generated above, we trained Bespoke-Stratos-32B, our reasoning model distilled from DeepSeek-R1 and using Berkeley NovaSky’s Sky-T1 data pipeline. The model outperforms Sky-T1 and o1-preview in reasoning (Math and Code) benchmarks, and almost reaches the performance of DeepSeek-R1-Distill-Qwen-32B while being trained on 47x fewer examples:

| Benchmark | Bespoke-Stratos-32B | Sky-T1-32B | o1-preview | DeepSeek-R1 (reported) | DeepSeek-R1-Distill-Qwen-32B (ours / reported) |
|-----------|--------------------:|------------:|-----------:|----------------------:|--------------------------------------------:|
| AIME2024 | 63.3 | 43.3 | 40.0 | 79.8 | 66.7 / 72.6 |
| MATH500 | 93.0 | 82.4 | 81.4 | 97.3 | 89.8 / 94.3 |
| GPQA-Diamond | 58.1 | 56.8 | 75.2 | 71.5 | 61.1 / 62.1 |
| LiveCodeBench v2 Easy | 96.7 | 86.3 | 92.9 | - | 91.2 / - |
| LiveCodeBench v2 Medium | 75.2 | 56.8 | 54.9 | - | 75.7 / - |
| LiveCodeBench v2 Hard | 26.2 | 17.9 | 16.3 | - | 38.2 / - |
| LiveCodeBench v2 All | 71.1 | 57.93 | 59.13 | - | 72.2 / - |

We open-source everything to continue experimenting together with the community!

- [32B Model](https://huggingface.co/bespokelabs/Bespoke-Stratos-32B) and [7B Model](https://huggingface.co/bespokelabs/Bespoke-Stratos-7B)
- [Reasoning Dataset](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)
- [Blog](https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation)

## Citation

```
@misc{
  bespoke_stratos, 
  author       = {Bespoke Labs},  
  title        = {Bespoke-Stratos: The unreasonable effectiveness of reasoning distillation},  
  howpublished = {https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation},  
  note         = {Accessed: 2025-01-22},  
  year         = {2025}
}
```
