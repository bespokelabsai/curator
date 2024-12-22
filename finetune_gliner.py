"""Script to finetune GLiNER model on PII detection dataset."""
import os
import json
import torch
import jieba
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set
from datasets import load_from_disk
from gliner import GLiNER, GLiNERConfig
from gliner.training.trainer import Trainer, TrainingArguments
from gliner.data_processing import DataCollator, WordsSplitter
from gliner.utils import load_config_as_namespace
from transformers import get_linear_schedule_with_warmup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gliner_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def validate_dataset(dataset) -> bool:
    """Validate that dataset has required fields and format."""
    try:
        # Check if dataset has required fields
        required_fields = {'input', 'output'}
        example = dataset[0]
        if not all(field in example for field in required_fields):
            logger.error(f"Dataset missing required fields. Found: {list(example.keys())}")
            return False
        
        # Validate output format
        if not isinstance(example['output'], list):
            logger.error("Output field must be a list of entities")
            return False
            
        # Validate entity format
        entity = example['output'][0]
        required_entity_fields = {'entity_type', 'entity_value', 'entity_source'}
        if not all(field in entity for field in required_entity_fields):
            logger.error(f"Entity missing required fields. Found: {list(entity.keys())}")
            return False
            
        # Validate entity values
        if not isinstance(entity['entity_value'], str):
            logger.error("Entity value must be a string")
            return False
            
        if not isinstance(entity['entity_type'], str):
            logger.error("Entity type must be a string")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Dataset validation failed: {str(e)}")
        return False

def load_and_prepare_data(model: GLiNER) -> Tuple[List[Dict], List[Dict], List[str]]:
    """Load dataset and convert to GLiNER format."""
    try:
        logger.info("Loading dataset from disk...")
        dataset = load_from_disk('pii_dataset')
        
        if not validate_dataset(dataset):
            raise ValueError("Dataset validation failed")
        
        logger.info(f"Dataset loaded successfully with {len(dataset)} examples")
        
        # Initialize GLiNER dataset
        gliner_data = []
        entity_types: Set[str] = set()
        entity_counts = {}
        
        # Map our entity types to GLiNER types
        type_mapping = {
            'Name': 'person',
            'Address': 'location',
            'Phone': 'phone',
            'Email': 'email',
            'Occupation': 'occupation'
        }
        
        logger.info("Converting dataset to GLiNER format...")
        for i, example in enumerate(dataset):
            try:
                text = example['input']
                
                # Tokenize text using whitespace first
                words = text.split()
                tokenized_text = []
                word_to_tokens = {}  # Map word index to token indices
                current_token_idx = 0
                
                # Tokenize each word and maintain mapping
                for word_idx, word in enumerate(words):
                    word_tokens = model.data_processor.transformer_tokenizer.tokenize(word)
                    word_to_tokens[word_idx] = (current_token_idx, current_token_idx + len(word_tokens) - 1)
                    tokenized_text.extend(word_tokens)
                    current_token_idx += len(word_tokens)
                
                # Process entities
                ner_spans = []
                for entity in example['output']:
                    entity_text = entity['entity_value']
                    entity_type = type_mapping.get(entity['entity_type'].title(), 'other')
                    entity_types.add(entity_type)
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                    
                    # Find entity words in text
                    entity_words = entity_text.split()
                    text_words = text.lower().split()
                    
                    for i in range(len(text_words) - len(entity_words) + 1):
                        if all(ew.lower() == tw for ew, tw in zip(entity_words, text_words[i:i+len(entity_words)])):
                            # Found entity, get token spans
                            start_token = word_to_tokens[i][0]
                            end_token = word_to_tokens[i + len(entity_words) - 1][1]
                            ner_spans.append([start_token, end_token, entity_type])
                            break
                
                # Only add examples with valid entities
                if ner_spans:
                    logger.info(f"Example {i}: Found {len(ner_spans)} entities")
                    gliner_data.append({
                        'tokenized_text': tokenized_text,
                        'ner': ner_spans
                    })
                else:
                    logger.warning(f"Example {i}: No valid entities found")
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1} examples...")
                    
            except Exception as e:
                logger.error(f"Error processing example {i}: {str(e)}")
                continue
        
        # Log entity statistics
        logger.info("Entity type distribution:")
        for entity_type, count in entity_counts.items():
            logger.info(f"  {entity_type}: {count}")
        
        # Split into train/test
        train_size = int(0.9 * len(gliner_data))
        train_dataset = gliner_data[:train_size]
        test_dataset = gliner_data[train_size:]
        
        logger.info(f"Split dataset into {len(train_dataset)} train and {len(test_dataset)} test examples")
        
        return train_dataset, test_dataset, list(entity_types)
        
    except Exception as e:
        logger.error(f"Failed to load and prepare data: {str(e)}")
        raise

def setup_training_environment() -> Tuple[GLiNER, torch.device, Path]:
    """Setup training environment and directories."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model and tokenizer
    logger.info("Initializing model and tokenizer...")
    from transformers import AutoTokenizer
    
    # Initialize GLiNER with configuration for Chinese text
    config = GLiNERConfig(
        model_name="bert-base-chinese",  # Use Chinese BERT as base model
        words_splitter_type="jieba",  # Use jieba for Chinese text
        max_len=256,  # Reduced from 384 to handle memory better
        max_types=25,
        hidden_size=768,  # Match BERT hidden size
        dropout=0.4,
        has_rnn=True,
        fine_tune=True
    )
    model = GLiNER(config=config)
    model.to(device)
    
    # Initialize tokenizer
    tokenizer = model.data_processor.transformer_tokenizer
    
    # Create output directories
    output_dir = Path("models/pii_detector")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return model, device, output_dir

def main():
    try:
        # Setup environment and model
        model, device, output_dir = setup_training_environment()
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        train_dataset, test_dataset, entity_types = load_and_prepare_data(model)
        logger.info(f"Found {len(entity_types)} entity types: {', '.join(entity_types)}")
        
        # Setup data collator
        logger.info("Setting up data collator...")
        data_collator = DataCollator(
            model.config,
            data_processor=model.data_processor,
            prepare_labels=True
        )
        
        # Calculate training parameters
        batch_size = 4  # Reduced batch size
        gradient_accumulation_steps = 4  # Add gradient accumulation
        effective_batch_size = batch_size * gradient_accumulation_steps
        num_steps = 500
        data_size = len(train_dataset)
        num_batches = data_size // effective_batch_size
        num_epochs = max(1, num_steps // num_batches)
        
        logger.info(f"Training parameters:")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Number of steps: {num_steps}")
        logger.info(f"  Number of epochs: {num_epochs}")
        
        # Setup training arguments with gradient accumulation and better logging
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=5e-6,
            weight_decay=0.01,
            others_lr=1e-5,
            others_weight_decay=0.01,
            focal_loss_alpha=0.75,
            focal_loss_gamma=2,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=0,
            max_steps=-1,  # No step limit
            save_steps=50,  # Save more frequently
            logging_steps=10,
            logging_first_step=True,
            max_grad_norm=1.0,
            warmup_ratio=0.1
        )
        
        # Initialize GLiNER trainer with custom training loop
        logger.info("Setting up trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        
        # Start training with GLiNER's training loop
        logger.info("Starting training...")
        try:
            # Initialize optimizer with different learning rates for different parameter groups
            optimizer = torch.optim.AdamW([
                {'params': model.model.parameters(), 'lr': trainer.args.learning_rate},
                {'params': model.linear.parameters(), 'lr': trainer.args.others_lr}
            ])
            trainer.optimizer = optimizer
            
            # Get dataloader and calculate steps
            train_dataloader = trainer.get_train_dataloader()
            num_update_steps_per_epoch = len(train_dataloader)
            total_steps = int(trainer.args.num_train_epochs * num_update_steps_per_epoch)
            warmup_steps = int(0.1 * total_steps)
            
            # Initialize learning rate scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            
            logger.info(f"Total training steps: {total_steps}")
            logger.info(f"Warmup steps: {warmup_steps}")
            
            # Training loop
            model.train()
            for epoch in range(int(trainer.args.num_train_epochs)):
                total_loss = 0
                model.train()
                
                for step, inputs in enumerate(train_dataloader):
                    try:
                        # Log input format for first batch
                        if epoch == 0 and step == 0:
                            logger.info("First batch input keys and shapes:")
                            for k, v in inputs.items():
                                if isinstance(v, torch.Tensor):
                                    logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                                else:
                                    logger.info(f"  {k}: type={type(v)}")
                        
                        # Move inputs to device
                        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in inputs.items()}
                        
                        # Forward pass and loss computation
                        try:
                            loss = trainer.training_step(model, inputs)
                            loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
                            if step == 0:
                                logger.info(f"Step {step}: Forward pass successful")
                        except Exception as e:
                            logger.error(f"Forward pass failed: {str(e)}")
                            raise
                        
                        # Backward pass
                        try:
                            loss.backward()
                            if step == 0:
                                logger.info(f"Step {step}: Backward pass successful")
                        except Exception as e:
                            logger.error(f"Backward pass failed: {str(e)}")
                            raise
                        
                        # Gradient clipping and optimization steps (only after accumulation)
                        if (step + 1) % gradient_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.args.max_grad_norm)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                        
                        total_loss += loss.item() * gradient_accumulation_steps  # Scale loss back for logging
                        
                        if step % 10 == 0:
                            avg_loss = total_loss / (step + 1)
                            lr = scheduler.get_last_lr()[0]
                            logger.info(f"Epoch {epoch+1}/{trainer.args.num_train_epochs}, "
                                      f"Step {step}/{num_update_steps_per_epoch}, "
                                      f"Loss: {avg_loss:.4f}, LR: {lr:.2e}")
                            
                    except Exception as e:
                        logger.error(f"Error in training step {step}: {str(e)}")
                        raise
                        
                    if trainer.args.max_steps > 0 and step >= trainer.args.max_steps:
                        break
                
                # Save checkpoint after each epoch
                if (epoch + 1) % trainer.args.save_steps == 0:
                    logger.info(f"Saving checkpoint for epoch {epoch+1}")
                    trainer.save_model(output_dir / f"checkpoint-{epoch+1}")
                    
            logger.info("Training completed!")
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise
        
        # Save final model
        final_model_path = output_dir / "final"
        model.save_pretrained(str(final_model_path))
        logger.info(f"Model saved to {final_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
