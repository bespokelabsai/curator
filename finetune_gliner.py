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
        
        max_len = model.config.max_len - 2  # Account for [CLS] and [SEP] tokens
        logger.info("Converting dataset to GLiNER format...")
        for i, example in enumerate(dataset):
            try:
                text = example['input']
                tokenizer = model.data_processor.transformer_tokenizer
                
                # Tokenize full text first to get total length
                full_tokens = tokenizer.tokenize(text)
                if len(full_tokens) > max_len:
                    logger.warning(f"Example {i}: Text length {len(full_tokens)} exceeds max_len {max_len}")
                
                # Process entities first to get their positions
                entities_info = []
                for entity in example['output']:
                    entity_text = entity['entity_value']
                    entity_type = type_mapping.get(entity['entity_type'].title(), 'other')
                    entity_types.add(entity_type)
                    
                    # Find entity position in original text
                    start_pos = text.find(entity_text)
                    if start_pos != -1:
                        end_pos = start_pos + len(entity_text)
                        entities_info.append({
                            'text': entity_text,
                            'type': entity_type,
                            'start': start_pos,
                            'end': end_pos
                        })
                
                if not entities_info:
                    logger.warning(f"Example {i}: No entities found in text")
                    continue
                
                # Sort entities by position
                entities_info.sort(key=lambda x: x['start'])
                
                # Find the text window that contains all entities
                first_entity_start = entities_info[0]['start']
                last_entity_end = entities_info[-1]['end']
                
                # Calculate context window
                context_size = (max_len - sum(len(tokenizer.tokenize(e['text'])) for e in entities_info)) // 2
                window_start = max(0, first_entity_start - context_size)
                window_end = min(len(text), last_entity_end + context_size)
                
                # Extract window text
                window_text = text[window_start:window_end]
                
                # Tokenize window text
                tokenized_text = tokenizer.tokenize(window_text)
                if len(tokenized_text) > max_len:
                    tokenized_text = tokenized_text[:max_len]
                
                # Adjust entity positions for window
                ner_spans = []
                current_pos = 0
                window_text_lower = window_text.lower()
                
                for entity in entities_info:
                    entity_text = entity['text']
                    entity_lower = entity_text.lower()
                    
                    # Find entity in window
                    entity_pos = window_text_lower.find(entity_lower, current_pos)
                    if entity_pos != -1:
                        # Get token positions
                        prefix_tokens = len(tokenizer.tokenize(window_text[:entity_pos]))
                        entity_tokens = len(tokenizer.tokenize(entity_text))
                        
                        # Only add if entity fits in window
                        if prefix_tokens + entity_tokens <= max_len:
                            ner_spans.append([
                                prefix_tokens,
                                prefix_tokens + entity_tokens - 1,
                                entity['type']
                            ])
                            entity_counts[entity['type']] = entity_counts.get(entity['type'], 0) + 1
                            current_pos = entity_pos + len(entity_text)
                
                # Only add examples with valid entities
                if ner_spans:
                    # Generate span indices for this example
                    seq_len = len(tokenized_text)
                    example_span_indices = []
                    span_mask = []
                    
                    # Generate spans more efficiently
                    max_width = model.config.max_width
                    min_width = getattr(model.config, 'min_width', 1)
                    
                    # First add spans for actual entities
                    entity_spans = set((start, end) for start, end, _ in ner_spans)
                    for start, end, _ in ner_spans:
                        example_span_indices.append([start, end])
                        span_mask.append(True)
                    
                    # Then add a limited number of negative samples around entities
                    for entity_start, entity_end, _ in ner_spans:
                        # Add spans before the entity (limited context)
                        context_start = max(0, entity_start - 2)  # Only 2 tokens before
                        for start in range(context_start, entity_start):
                            for width in range(min_width, min(3, entity_start - start + 1)):
                                end = start + width - 1
                                if (start, end) not in entity_spans:
                                    example_span_indices.append([start, end])
                                    span_mask.append(False)
                        
                        # Add spans after the entity (limited context)
                        context_end = min(seq_len, entity_end + 2)  # Only 2 tokens after
                        for start in range(entity_end + 1, context_end):
                            for width in range(min_width, min(3, context_end - start + 1)):
                                end = start + width - 1
                                if end < seq_len and (start, end) not in entity_spans:
                                    example_span_indices.append([start, end])
                                    span_mask.append(False)
                    
                    # Ensure we have at least some spans
                    if not example_span_indices:
                        # Add some basic spans if no entities found
                        for start in range(0, min(seq_len, 3)):
                            end = start  # Single token spans
                            example_span_indices.append([start, end])
                            span_mask.append(False)
                            
                    # Limit total number of spans
                    max_spans_per_example = 100
                    if len(example_span_indices) > max_spans_per_example:
                        example_span_indices = example_span_indices[:max_spans_per_example]
                        span_mask = span_mask[:max_spans_per_example]
                    
                    # Pad spans to fixed size if needed
                    max_spans = seq_len * model.config.max_width
                    if len(example_span_indices) < max_spans:
                        padding = [[0, 0]] * (max_spans - len(example_span_indices))
                        example_span_indices.extend(padding)
                        span_mask.extend([False] * (max_spans - len(span_mask)))
                    
                    # Create span labels for each possible span
                    num_spans = len(example_span_indices)
                    entity_type_to_idx = {label: idx for idx, label in enumerate(sorted(entity_types))}
                    num_labels = len(entity_types)
                    span_labels = []
                    
                    # Generate labels for each span
                    for span_idx, (start, end) in enumerate(example_span_indices[:num_spans]):
                        # Initialize label vector
                        label_vector = [0] * num_labels
                        # Check if this span matches any entity
                        for entity_start, entity_end, entity_type in ner_spans:
                            if start == entity_start and end == entity_end:
                                label_vector[entity_type_to_idx[entity_type]] = 1
                        span_labels.append(label_vector)
                    
                    # Pad span labels if needed
                    if len(span_labels) < max_spans:
                        padding = [[0] * num_labels] * (max_spans - len(span_labels))
                        span_labels.extend(padding)
                    
                    # Convert to exact GLiNER format with proper tokenization
                    # Tokenize text with BERT tokenizer
                    tokenizer = model.data_processor.transformer_tokenizer
                    encoding = tokenizer(
                        window_text,
                        padding='max_length',
                        truncation=True,
                        max_length=model.config.max_len,
                        return_tensors='pt'
                    )
                    
                    # Extract and convert tensors to lists
                    input_ids = encoding['input_ids'][0].tolist()
                    attention_mask = encoding['attention_mask'][0].tolist()
                    
                    # Convert token indices to match input_ids
                    token_spans = []
                    for start, end, label in ner_spans:
                        # Get subword tokens for the entity
                        entity_text = window_text[start:end]
                        entity_tokens = tokenizer.encode(entity_text, add_special_tokens=False)
                        # Find these tokens in the full encoding
                        for i in range(len(input_ids)):
                            if i + len(entity_tokens) <= len(input_ids):
                                if input_ids[i:i+len(entity_tokens)] == entity_tokens:
                                    token_spans.append([i, i+len(entity_tokens)-1, label])
                                    break
                    
                    tokenized_example = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'tokenized_text': tokenizer.convert_ids_to_tokens(input_ids),
                        'ner': token_spans,
                        'span_indices': example_span_indices,
                        'span_labels': span_labels,
                        'span_mask': span_mask,
                        'num_spans': num_spans
                    }
                    logger.info(f"Example {i}: Found {len(ner_spans)} entities, generated {len(example_span_indices)} spans")
                    logger.info(f"First entity span: {ner_spans[0] if ner_spans else 'None'}")
                    gliner_data.append(tokenized_example)
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
        fine_tune=True,
        max_width=10,  # Reduced maximum span width to prevent excessive spans
        span_mode="marker",  # Use marker-based span representation
        min_width=1  # Add minimum width to filter out invalid spans
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
        
        # Log example data format and tensor shapes
        if train_dataset:
            example = train_dataset[0]
            logger.info("\nData format analysis:")
            logger.info("=" * 50)
            logger.info(f"Keys: {list(example.keys())}")
            logger.info(f"Tokenized text length: {len(example['tokenized_text'])}")
            logger.info(f"Number of entities: {len(example['ner'])}")
            logger.info(f"First entity: {example['ner'][0] if example['ner'] else 'None'}")
            logger.info(f"Number of spans: {example['num_spans']}")
            logger.info(f"Span indices shape: {len(example['span_indices'])}x2")
            logger.info(f"Span labels shape: {len(example['span_labels'])}x{len(example['span_labels'][0])}")
            logger.info(f"Span mask length: {len(example['span_mask'])}")
            logger.info("=" * 50)
        
        # Setup data collator with span processing
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
            # Initialize optimizer with single learning rate and gradient clipping
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=trainer.args.learning_rate,
                weight_decay=trainer.args.weight_decay,
                eps=1e-8
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
