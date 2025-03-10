import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

MODEL_NAME = "unsloth/llama-3-8b-Instruct"
OUTPUT_DIR = "./llama3-finetuned"
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-5 * GRADIENT_ACCUMULATION_STEPS
MAX_LENGTH = 8192
TEST_SIZE = 0.05
USE_DEEPSPEED = os.getenv("DISTRIBUTED", False)
DS_CONFIG_PATH = "ds_config.json"
DATA_PATH = "./raft_dataset.parquet"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def process_dataset(data_path):
    """Load and preprocess the dataset from a JSON file or directory of examples."""
    ds = load_dataset("parquet", data_files=data_path)
    return ds["train"]


print("Processing dataset...")
dataset = process_dataset(DATA_PATH)


def format_for_llama(example):
    """Format examples as instructions for LLaMA 3.1 (Unsloth) with correct special tokens."""
    return {"text": f"<|begin_of_text|>[INST] {example['instruction']} [/INST] <|eot_id|> {example['cot_answer']} "}


print("Formatting dataset for Llama 3...")
formatted_dataset = dataset.map(format_for_llama, remove_columns=[col for col in dataset.column_names if col != "text"])

print(f"Creating train/validation split with {TEST_SIZE*100}% validation data...")
splits = formatted_dataset.train_test_split(test_size=TEST_SIZE)
train_dataset = splits["train"]
eval_dataset = splits["test"]

print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")


# Tokenize the datasets
def tokenize_function(examples):
    """Tokenize the examples with the LLaMA 3 tokenizer."""
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


print("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)

tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    evaluation_strategy="steps",
    eval_steps=3,
    save_strategy="steps",
    save_steps=9,
    save_total_limit=3,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    bf16=True,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=1,
    deepspeed=DS_CONFIG_PATH if USE_DEEPSPEED else None,
    load_best_model_at_end=True,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting Full fine-tuning...")
trainer.train()

model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"Model saved to {OUTPUT_DIR}/final")
