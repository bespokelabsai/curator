"""Script to finetune GLiNER model on PII detection dataset."""
import os
import re
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

# Initialize jieba for Chinese text segmentation
jieba.initialize()

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

def normalize_text(text: str) -> str:
    """Normalize text for better matching."""
    # Remove extra whitespace and lowercase for English text
    text = ' '.join(text.lower().split())
    # Remove spaces between Chinese characters
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
    return text

def load_and_prepare_data(model: GLiNER) -> Tuple[List[Dict], List[Dict], List[str]]:
    """Load dataset and convert to GLiNER format with proper entity alignment."""
    # Map our entity types to GLiNER types
    type_mapping = {
        'Name': 'person',
        'Address': 'location',
        'Phone': 'phone',
        'Email': 'email',
        'Occupation': 'occupation',
        'Company': 'organization',
        'Date': 'date',
        'ID': 'misc'
    }
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
        
        # Initialize GLiNER dataset
        gliner_data = []
        entity_types: Set[str] = set()
        entity_counts = {}
        
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
                    
                    # Generate spans with guaranteed minimum count
                    max_width = min(model.config.max_width, 5)  # Limit max width to prevent explosion
                    min_width = 1  # Always use 1 as min width
                    min_spans = 10  # Guarantee at least this many spans
                    
                    # First add spans for actual entities
                    entity_spans = set((start, end) for start, end, _ in ner_spans)
                    for start, end, _ in ner_spans:
                        if end >= start:  # Validate span
                            example_span_indices.append([start, end])
                            span_mask.append(True)
                    
                    # Generate sliding window spans
                    for start in range(seq_len):
                        for width in range(min_width, min(max_width + 1, seq_len - start + 1)):
                            end = start + width - 1
                            if end < seq_len:
                                span = (start, end)
                                if span not in entity_spans:
                                    example_span_indices.append([start, end])
                                    span_mask.append(False)
                                    
                                if len(example_span_indices) >= 100:  # Limit total spans
                                    break
                        if len(example_span_indices) >= 100:
                            break
                    
                    # If we don't have enough spans, add single-token spans
                    while len(example_span_indices) < min_spans:
                        for start in range(min(seq_len, min_spans)):
                            if len(example_span_indices) >= min_spans:
                                break
                            span = [start, start]
                            if span not in example_span_indices:
                                example_span_indices.append(span)
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
                        # Find these tokens in the full encoding with robust matching
                        found = False
                        for i in range(len(input_ids)):
                            if i + len(entity_tokens) <= len(input_ids):
                                current_span = input_ids[i:i+len(entity_tokens)]
                                # Try exact match
                                if current_span == entity_tokens:
                                    token_spans.append([i, i+len(entity_tokens)-1, label])
                                    found = True
                                    break
                                # Try partial match for subword tokens
                                elif len(current_span) > 0 and len(entity_tokens) > 0:
                                    # Decode and compare text
                                    span_text = tokenizer.decode(current_span)
                                    entity_text = tokenizer.decode(entity_tokens)
                                    if span_text.lower().strip() == entity_text.lower().strip():
                                        token_spans.append([i, i+len(entity_tokens)-1, label])
                                        found = True
                                        break
                        
                        if not found:
                            logger.debug(f"Could not find token span for entity: {entity['text']}")
                    
                    # Convert data to tensors while preserving required fields
                    tokenized_text = tokenizer.convert_ids_to_tokens(input_ids)
                    # Create position-aware text lengths
                    seq_length = len(input_ids)
                    text_lengths = torch.arange(1, seq_length + 1, dtype=torch.long)
                    
                    tokenized_example = {
                        'input_ids': torch.tensor(input_ids, dtype=torch.long),
                        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                        'token_type_ids': torch.zeros_like(torch.tensor(input_ids, dtype=torch.long)),
                        'words_mask': torch.tensor(attention_mask, dtype=torch.long),
                        'span_indices': torch.tensor(example_span_indices, dtype=torch.long),
                        'span_labels': torch.tensor(span_labels, dtype=torch.float),
                        'span_mask': torch.tensor(span_mask, dtype=torch.bool),
                        'text_lengths': text_lengths,
                        'tokenized_text': tokenized_text,  # Keep for debugging
                        'ner': token_spans  # Keep original NER annotations
                    }
                    
                    # Validate tensor shapes
                    B = 1  # Single example
                    L = len(input_ids)
                    S = len(example_span_indices)
                    T = len(entity_types)
                    
                    assert tokenized_example['input_ids'].shape == (L,), f"Expected input_ids shape (L,), got {tokenized_example['input_ids'].shape}"
                    assert tokenized_example['attention_mask'].shape == (L,), f"Expected attention_mask shape (L,), got {tokenized_example['attention_mask'].shape}"
                    assert tokenized_example['span_indices'].shape == (S, 2), f"Expected span_indices shape (S,2), got {tokenized_example['span_indices'].shape}"
                    assert tokenized_example['span_labels'].shape == (S, T), f"Expected span_labels shape (S,T), got {tokenized_example['span_labels'].shape}"
                    assert tokenized_example['span_mask'].shape == (S,), f"Expected span_mask shape (S,), got {tokenized_example['span_mask'].shape}"
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
    
    # Define our entity types
    entity_types = ['person', 'location', 'phone', 'email', 'occupation']
    id2label = {i: label for i, label in enumerate(entity_types)}
    label2id = {label: i for i, label in id2label.items()}
    
    logger.info(f"Initializing model with entity types: {entity_types}")
    
    # Initialize GLiNER with configuration for Chinese text
    config = GLiNERConfig(
        model_name="bert-base-chinese",  # Use Chinese BERT as base model
        words_splitter_type="jieba",  # Use jieba for Chinese text
        max_len=256,  # Reduced from 384 to handle memory better
        max_types=len(entity_types),  # Match our entity types count
        hidden_size=768,  # Match BERT hidden size
        dropout=0.1,  # Reduced dropout for stability
        has_rnn=False,  # Disable RNN to simplify architecture
        fine_tune=True,
        max_width=5,  # Reduced max width for stability
        span_mode="marker",  # Use marker-based span representation
        min_width=1,  # Minimum span width
        use_focal_loss=True,  # Enable focal loss for imbalanced classes
        focal_loss_gamma=2.0,  # Focal loss gamma parameter
        focal_loss_alpha=0.25,  # Focal loss alpha parameter
        labels_encoder="bert-base-chinese",  # Use same model for labels encoding
        id2label=id2label,
        label2id=label2id,
        use_prompt=False,  # Disable prompt-based learning
        use_type_embeddings=True  # Enable type embeddings for better entity typing
    )
    
    # Initialize model with proper configuration
    model = GLiNER(config=config)
    model = model.to(device)
    
    # Print model configuration
    logger.info("Model configuration:")
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")
    
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
            
            # Log tensor shapes and sizes
            for key, value in example.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"{key}: shape={value.shape}, dtype={value.dtype}")
                    # Check for NaN or Inf values
                    if torch.isnan(value).any():
                        logger.error(f"NaN values found in {key}")
                    if torch.isinf(value).any():
                        logger.error(f"Inf values found in {key}")
                elif isinstance(value, list):
                    if key == 'tokenized_text':
                        logger.info(f"{key}: length={len(value)}")
                        logger.info(f"First 10 tokens: {value[:10]}")
                    elif key == 'ner':
                        logger.info(f"{key}: {len(value)} entities")
                        if value:
                            logger.info(f"First entity: {value[0]}")
                else:
                    logger.info(f"{key}: type={type(value)}")
            logger.info("=" * 50)
            
            # Validate required fields for GLiNER
            required_fields = {'tokenized_text', 'ner'}
            missing_fields = required_fields - set(example.keys())
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Setup custom data collator with proper batching
        logger.info("Setting up data collator...")
        class CustomDataCollator:
            def __init__(self, config, data_processor, entity_types, prepare_labels=True):
                self.config = config
                self.data_processor = data_processor
                self.entity_types = entity_types
                self.prepare_labels = prepare_labels
                logger.info(f"Initialized CustomDataCollator with {len(entity_types)} entity types: {entity_types}")
                
            def __call__(self, features):
                batch = {}
                
                # Get max sequence length in batch
                max_length = max(len(feature['input_ids']) for feature in features)
                max_length = min(max_length, self.config.max_len)
                
                # Get max number of spans in batch
                max_spans = max(feature['span_indices'].size(0) for feature in features)
                
                # Initialize tensors for batch
                batch_size = len(features)
                batch['input_ids'] = torch.zeros((batch_size, max_length), dtype=torch.long)
                batch['attention_mask'] = torch.zeros((batch_size, max_length), dtype=torch.long)
                batch['token_type_ids'] = torch.zeros((batch_size, max_length), dtype=torch.long)
                batch['words_mask'] = torch.zeros((batch_size, max_length), dtype=torch.long)
                batch['span_indices'] = torch.zeros((batch_size, max_spans, 2), dtype=torch.long)
                batch['span_labels'] = torch.zeros((batch_size, max_spans, len(self.entity_types)), dtype=torch.float)
                batch['span_mask'] = torch.zeros((batch_size, max_spans), dtype=torch.bool)
                batch['text_lengths'] = torch.zeros((batch_size, max_length), dtype=torch.long)
                
                # Fill batch tensors
                for i, feature in enumerate(features):
                    seq_length = len(feature['input_ids'])
                    num_spans = feature['span_indices'].size(0)
                    
                    # Sequence tensors
                    batch['input_ids'][i, :seq_length] = feature['input_ids']
                    batch['attention_mask'][i, :seq_length] = feature['attention_mask']
                    batch['token_type_ids'][i, :seq_length] = feature['token_type_ids']
                    batch['words_mask'][i, :seq_length] = feature['words_mask']
                    # Set text_lengths to match sequence length for each position
                    batch['text_lengths'][i, :seq_length] = torch.arange(1, seq_length + 1)
                    
                    # Span tensors
                    batch['span_indices'][i, :num_spans] = feature['span_indices']
                    batch['span_labels'][i, :num_spans] = feature['span_labels']
                    batch['span_mask'][i, :num_spans] = feature['span_mask']
                
                # Move to device if needed
                device = next(iter(features[0].values())).device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Validate tensor shapes
                logger.info(f"Batch shapes:")
                for k, v in batch.items():
                    logger.info(f"  {k}: {v.shape}")
                
                return batch
        
        data_collator = CustomDataCollator(
            model.config,
            data_processor=model.data_processor,
            entity_types=entity_types,
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
                        
                        # Move inputs to device and ensure proper shapes
                        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in inputs.items()}
                        
                        # Log tensor shapes for debugging
                        if epoch == 0 and step == 0:
                            logger.info("\nInput tensor shapes after moving to device:")
                            for k, v in inputs.items():
                                if isinstance(v, torch.Tensor):
                                    logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                                    # Check for NaN or inf values
                                    if torch.isnan(v).any():
                                        logger.warning(f"NaN values detected in {k}")
                                    if torch.isinf(v).any():
                                        logger.warning(f"Inf values detected in {k}")
                        
                        # Log input shapes before forward pass
                        if step == 0:
                            logger.info("\nInput shapes before forward pass:")
                            for k, v in inputs.items():
                                if isinstance(v, torch.Tensor):
                                    logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                                    if torch.isnan(v).any():
                                        logger.warning(f"NaN values detected in {k}")
                                    if torch.isinf(v).any():
                                        logger.warning(f"Inf values detected in {k}")
                        
                        # Forward pass with detailed error handling
                        try:
                            # Forward pass and loss calculation
                            model_inputs = {
                                'input_ids': inputs['input_ids'],
                                'attention_mask': inputs['attention_mask'],
                                'token_type_ids': inputs['token_type_ids'],
                                'span_indices': inputs['span_indices'],
                                'span_labels': inputs['span_labels'],
                                'span_mask': inputs['span_mask'],
                                'words_mask': inputs['words_mask'],
                                'text_lengths': inputs['text_lengths']
                            }
                            # Prepare label inputs for bi-encoder with improved error handling
                            tokenizer = model.data_processor.transformer_tokenizer
                            entity_texts = []
                            
                            try:
                                for i in range(inputs['span_labels'].size(0)):
                                    batch_texts = []
                                    for j in range(inputs['span_labels'].size(1)):
                                        if inputs['span_mask'][i, j]:
                                            # Get label index with bounds checking
                                            label_tensor = inputs['span_labels'][i, j]
                                            if label_tensor.dim() == 0:
                                                continue
                                            label_idx = label_tensor.argmax().item()
                                            
                                            # Safely get entity type
                                            id2label = list(model.config.id2label.values())
                                            if 0 <= label_idx < len(id2label):
                                                entity_type = id2label[label_idx]
                                            else:
                                                logger.warning(f"Invalid label index {label_idx}, skipping")
                                                continue
                                            
                                            # Safely get text span
                                            start, end = inputs['span_indices'][i, j]
                                            if not (0 <= start <= end < inputs['input_ids'].size(1)):
                                                logger.warning(f"Invalid span indices ({start}, {end}), skipping")
                                                continue
                                            
                                            # Get entity text
                                            text_tokens = inputs['input_ids'][i, start:end+1]
                                            entity_text = tokenizer.decode(text_tokens)
                                            if entity_text.strip():  # Only add non-empty texts
                                                batch_texts.append(f"{entity_type}: {entity_text}")
                                    
                                    # Add batch texts or placeholder
                                    entity_texts.extend(batch_texts if batch_texts else ["O"])
                            except Exception as e:
                                logger.error(f"Error processing entity texts: {str(e)}")
                                entity_texts = ["O"] * inputs['span_labels'].size(0)
                            
                            # Join entity texts with [SEP] token, ensure at least one entry per batch
                            labels_text = " [SEP] ".join(entity_texts)
                            
                            # Tokenize labels text
                            labels_tokens = tokenizer(
                                labels_text,
                                padding='max_length',
                                truncation=True,
                                max_length=128,
                                return_tensors='pt'
                            )
                            
                            # Move label tensors to device
                            labels_input_ids = labels_tokens['input_ids'].to(device)
                            labels_attention_mask = labels_tokens['attention_mask'].to(device)
                            
                            # Forward pass with bi-encoder inputs and error handling
                            try:
                                outputs = model(
                                    input_ids=inputs['input_ids'],
                                    attention_mask=inputs['attention_mask'],
                                    token_type_ids=inputs['token_type_ids'],
                                    labels_input_ids=labels_input_ids,
                                    labels_attention_mask=labels_attention_mask,
                                    span_indices=inputs['span_indices'],
                                    span_labels=inputs['span_labels'],
                                    span_mask=inputs['span_mask'],
                                    words_mask=inputs['words_mask'],
                                    text_lengths=inputs['text_lengths']
                                )
                                
                                # Validate outputs
                                if outputs is None:
                                    raise ValueError("Model returned None outputs")
                                if not hasattr(outputs, 'loss'):
                                    raise ValueError("Model outputs missing loss attribute")
                                
                            except Exception as e:
                                logger.error(f"Forward pass failed with error: {str(e)}")
                                continue
                            
                            if outputs is None or not hasattr(outputs, 'loss'):
                                logger.error("Model outputs invalid or missing loss")
                                continue
                            
                            loss = outputs.loss
                            if loss is None:
                                logger.error("Model returned None loss")
                                continue
                                
                            if torch.isnan(loss):
                                logger.error("Loss is NaN, skipping batch")
                                continue
                            if torch.isinf(loss):
                                logger.error("Loss is Inf, skipping batch")
                                continue
                                
                            # Scale loss for gradient accumulation
                            loss = loss / gradient_accumulation_steps
                            if step % 10 == 0:
                                logger.info(f"Step {step}: Loss before scaling = {loss.item() * gradient_accumulation_steps:.4f}")
                        except Exception as e:
                            logger.error(f"Forward pass failed: {str(e)}")
                            logger.error("Skipping problematic batch")
                            continue
                        
                        # Backward pass with gradient monitoring
                        try:
                            loss.backward()
                            
                            # Monitor gradients
                            if step % 10 == 0:
                                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.args.max_grad_norm)
                                logger.info(f"Step {step}: Gradient norm = {grad_norm:.4f}")
                        except Exception as e:
                            logger.error(f"Backward pass failed: {str(e)}")
                            logger.error("Skipping problematic batch")
                            continue
                        
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
