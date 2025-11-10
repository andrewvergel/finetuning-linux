"""Data processing utilities for the finetuning project."""
import json
import logging
from typing import Dict, List, Optional, Union, Any
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data loading and preprocessing for model training and evaluation."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int = 1024):
        """Initialize the data processor.
        
        Args:
            tokenizer: The tokenizer to use for text processing
            max_seq_length: Maximum sequence length for truncation
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_dataset(self, file_path: str, split: str = "train") -> Dataset:
        """Load dataset from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            split: Dataset split name (train/validation/test)
            
        Returns:
            Loaded dataset
        """
        try:
            dataset = load_dataset('json', data_files={split: file_path}, split=split)
            logger.info(f"Loaded dataset from {file_path} with {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset from {file_path}: {e}")
            raise
    
    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess the dataset for training.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Preprocessed dataset
        """
        # Filter out invalid examples
        dataset = dataset.filter(
            self._validate_example,
            batched=False,
            desc="Validating examples"
        )
        
        # Apply chat template formatting
        dataset = dataset.map(
            self._format_example,
            batched=False,
            desc="Formatting examples"
        )
        
        return dataset
    
    def _validate_example(self, example: Dict[str, Any]) -> bool:
        """Validate that an example has the required fields."""
        required_fields = ["input", "output"]
        return all(field in example and example[field].strip() for field in required_fields)
    
    def _format_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Format an example using the chat template."""
        system = example.get("system", "Eres un asistente útil y conciso.")
        user = example["input"]
        assistant = example["output"]
        
        # Format messages according to the chat template
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False  # We include the assistant's response in training
        )
        
        # Ensure EOS token is present
        eos_token = self.tokenizer.eos_token or "</s>"
        if not text.endswith(eos_token):
            text += eos_token
            
        return {"text": text, "length": len(text)}
    
    def tokenize_function(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Tokenize the examples."""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_token_type_ids=False,
        )
    
    def prepare_for_training(
        self, 
        train_file: str, 
        val_file: Optional[str] = None,
        test_size: float = 0.1,
        seed: int = 42
    ) -> Dict[str, Dataset]:
        """Prepare datasets for training.
        
        Args:
            train_file: Path to training data file
            val_file: Optional path to validation data file
            test_size: Fraction of training data to use for validation if val_file is None
            seed: Random seed for splitting
            
        Returns:
            Dictionary with 'train' and optionally 'validation' datasets
        """
        # Load training data
        train_dataset = self.load_dataset(train_file, "train")
        train_dataset = self.preprocess_dataset(train_dataset)
        
        # Load or create validation data
        if val_file:
            val_dataset = self.load_dataset(val_file, "validation")
            val_dataset = self.preprocess_dataset(val_dataset)
        else:
            # Split training data if no validation file is provided
            split = train_dataset.train_test_split(test_size=test_size, seed=seed)
            train_dataset = split["train"]
            val_dataset = split["test"]
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing training data"
        )
        
        tokenized_val = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation data"
        )
        
        return {
            "train": tokenized_train,
            "validation": tokenized_val
        }
