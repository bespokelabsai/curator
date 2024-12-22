"""Script to finetune GLiNER model on PII detection dataset."""
import os
import json
import torch
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set
from datasets import load_from_disk
from gliner import GLiNER, GLiNERConfig
from gliner.training.trainer import Trainer, TrainingArguments
from gliner.data_processing import DataCollator, WordsSplitter
from gliner.utils import load_config_as_namespace

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
        
        logger.info("Converting dataset to GLiNER format...")
        for i, example in enumerate(dataset):
            try:
                # Map our entity types to GLiNER types
                type_mapping = {
                    'Name': 'person',
                    'Address': 'location',
                    'Phone': 'phone',
                    'Email': 'email',
                    'Occupation': 'occupation'
                }
                
                text = example['input']
                # Get tokenized text and offsets using the model's tokenizer
                encoding = model.data_processor.transformer_tokenizer(
                    text, 
                    return_offsets_mapping=True, 
                    add_special_tokens=False
                )
                tokens = encoding.tokens()
                offset_mapping = encoding.offset_mapping
                
                # Process entities and convert to token spans
                ner_spans = []
                for entity in example['output']:
                    entity_text = entity['entity_value']
                    entity_type = type_mapping.get(entity['entity_type'].title(), 'other')
                    entity_types.add(entity_type)
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                    
                    # Find entity in text
                    start = text.find(entity_text)
                    if start != -1:
                        end = start + len(entity_text)
                        
                        # Find token spans that contain the entity
                        token_start = None
                        token_end = None
                        for idx, (token_start_char, token_end_char) in enumerate(offset_mapping):
                            if token_start is None and token_start_char <= start < token_end_char:
                                token_start = idx
                            if token_end is None and token_start_char < end <= token_end_char:
                                token_end = idx
                                break
                        
                        if token_start is not None and token_end is not None:
                            ner_spans.append([token_start, token_end, entity_type])
                
                # Only add examples with valid entities
                if ner_spans:
                    logger.info(f"Example {i}: Found {len(ner_spans)} entities")
                    gliner_data.append({
                        'tokenized_text': tokens,
                        'ner': ner_spans
                    })
                else:
                    logger.warning(f"Example {i}: Skipping - no valid entities found")
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1} examples...")
                    
            except Exception as e:
                logger.warning(f"Error processing example {i}: {str(e)}")
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
    
    # Initialize GLiNER with its default configuration
    model = GLiNER.from_pretrained("urchade/gliner_small", trust_remote_code=True)
    # Use the model's data processor tokenizer
    tokenizer = model.data_processor.transformer_tokenizer
    model.data_processor.transformer_tokenizer = tokenizer
    model.to(device)
    
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
        batch_size = 8
        num_steps = 500
        data_size = len(train_dataset)
        num_batches = data_size // batch_size
        num_epochs = max(1, num_steps // num_batches)
        
        logger.info(f"Training parameters:")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Number of steps: {num_steps}")
        logger.info(f"  Number of epochs: {num_epochs}")
        
        # Setup training arguments
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
            dataloader_num_workers=0
        )
        
        # Initialize GLiNER trainer
        logger.info("Setting up trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed!")
        
        # Save final model
        final_model_path = output_dir / "final"
        model.save_pretrained(str(final_model_path))
        logger.info(f"Model saved to {final_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
