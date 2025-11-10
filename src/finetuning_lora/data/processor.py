"""Data processing utilities for LoRA fine-tuning."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import datasets
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from finetuning_lora.config.data import DataConfig

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles loading and processing of training data."""
    
    def __init__(self, config: DataConfig, tokenizer: Optional[PreTrainedTokenizerBase] = None):
        """Initialize the data processor.
        
        Args:
            config: Data configuration
            tokenizer: Optional tokenizer for text processing
        """
        self.config = config
        self.tokenizer = tokenizer
        self._validate_config()
        
    def _validate_config(self):
        """Validate the data configuration."""
        if not Path(self.config.train_path).exists():
            raise FileNotFoundError(f"Training data not found at {self.config.train_path}")
            
        if self.config.val_path and not Path(self.config.val_path).exists():
            logger.warning(f"Validation data not found at {self.config.val_path}, will split from training")
            
    def load_datasets(self) -> DatasetDict:
        """Load and split the dataset.
        
        Returns:
            DatasetDict containing train and validation splits
        """
        # Load training data
        train_data = self._load_jsonl(self.config.train_path)
        train_dataset = Dataset.from_list(train_data)
        
        # Handle validation data
        if self.config.val_path:
            val_data = self._load_jsonl(self.config.val_path)
            val_dataset = Dataset.from_list(val_data)
        else:
            # Split training data if no validation file provided
            split = train_dataset.train_test_split(
                test_size=self.config.validation_split,
                seed=self.config.shuffle_seed
            )
            train_dataset = split["train"]
            val_dataset = split["test"]
            
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
        
    def _load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load examples from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of examples
        """
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    example = json.loads(line.strip())
                    if self._validate_example(example):
                        examples.append(example)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {file_path}: {e}")
        return examples
        
    def _validate_example(self, example: Dict[str, Any]) -> bool:
        """Validate that an example has the correct structure.
        
        Args:
            example: Example to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(example, dict):
            return False
            
        required_fields = ["input", "output"]
        return all(field in example and example.get(field) for field in required_fields)
    
    def format_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Format an example using the chat template.
        
        Args:
            example: Input example with 'input' and 'output' keys, optionally 'system'
            
        Returns:
            Formatted example with 'text' key
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer is required for formatting examples")
            
        if not self._validate_example(example):
            raise ValueError(f"Invalid example format: {example}")
            
        # Build messages list with optional system message
        messages = []
        
        # Add system message if present
        if "system" in example and example["system"]:
            messages.append({"role": "system", "content": example["system"]})
        
        # Add user and assistant messages
        messages.append({"role": "user", "content": example["input"]})
        messages.append({"role": "assistant", "content": example["output"]})
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Ensure text ends with EOS token
        eos_token = self.tokenizer.eos_token or "</s>"
        if not text.endswith(eos_token):
            text += eos_token
        
        return {"text": text}
    
    def tokenize_function(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Tokenize a batch of examples.
        
        Args:
            examples: Batch of examples with 'text' key
            
        Returns:
            Tokenized batch
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer is required for tokenization")
            
        tokenized = self.tokenizer(
            examples["text"],
            **self.config.get_tokenizer_kwargs(),
            padding="max_length" if self.config.padding_side == "right" else False,
            truncation=True,
            return_tensors=None,
            add_special_tokens=True
        )
        
        # For left padding
        if self.config.padding_side == "left":
            tokenized["input_ids"] = [
                [self.tokenizer.pad_token_id] * (self.config.max_length - len(seq)) + seq
                for seq in tokenized["input_ids"]
            ]
            tokenized["attention_mask"] = [
                [0] * (self.config.max_length - len(seq)) + seq
                for seq in tokenized["attention_mask"]
            ]
            
        return tokenized
    
    def prepare_datasets(self, tokenizer: PreTrainedTokenizerBase) -> Tuple[Dataset, Dataset]:
        """Prepare the training and validation datasets.
        
        Args:
            tokenizer: Tokenizer to use for processing
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        self.tokenizer = tokenizer
        
        # Load and split the data
        datasets = self.load_datasets()
        
        # Format the examples
        train_dataset = datasets["train"].map(
            self.format_example,
            remove_columns=datasets["train"].column_names,
            desc="Formatting training examples"
        )
        
        val_dataset = datasets["validation"].map(
            self.format_example,
            remove_columns=datasets["validation"].column_names,
            desc="Formatting validation examples"
        )
        
        # Tokenize
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing training data"
        )
        
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing validation data"
        )
        
        return train_dataset, val_dataset
    
    def prepare_datasets_for_sft(
        self, 
        tokenizer: PreTrainedTokenizerBase,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """Prepare datasets for SFTTrainer (with text field, not tokenized).
        
        Args:
            tokenizer: Tokenizer to use for formatting (chat template)
            shuffle: Whether to shuffle the datasets
            seed: Random seed for shuffling
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        self.tokenizer = tokenizer
        
        # Load and split the data
        datasets = self.load_datasets()
        
        # Format the examples (creates "text" field with chat template)
        train_dataset = datasets["train"].map(
            self.format_example,
            remove_columns=datasets["train"].column_names,
            desc="Formatting training examples"
        )
        
        val_dataset = datasets["validation"].map(
            self.format_example,
            remove_columns=datasets["validation"].column_names,
            desc="Formatting validation examples"
        ) if len(datasets["validation"]) > 0 else None
        
        # Shuffle if requested
        if shuffle:
            shuffle_seed = seed or self.config.shuffle_seed
            train_dataset = train_dataset.shuffle(seed=shuffle_seed)
            if val_dataset is not None:
                val_dataset = val_dataset.shuffle(seed=shuffle_seed)
        
        return train_dataset, val_dataset
