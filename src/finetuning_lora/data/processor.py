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
    """Handles loading and processing of training data for LoRA fine-tuning.
    
    This class provides functionality to:
    - Load data from JSONL files
    - Validate data format (requires 'input' and 'output' fields, optional 'system' field)
    - Format examples using chat templates
    - Prepare datasets for SFTTrainer
    
    The processor handles both dictionary objects and HuggingFace Dataset Row objects,
    making it compatible with different data loading scenarios.
    
    Example:
        >>> from finetuning_lora.config.data import DataConfig
        >>> from transformers import AutoTokenizer
        >>> 
        >>> config = DataConfig(train_path="data/instructions.jsonl")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        >>> processor = DataProcessor(config, tokenizer)
        >>> train_dataset, val_dataset = processor.prepare_datasets_for_sft(tokenizer)
    """
    
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
            List of valid examples
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is empty or contains no valid examples
        """
        examples = []
        line_number = 0
        invalid_count = 0
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        example = json.loads(line)
                        if self._validate_example(example):
                            examples.append(example)
                        else:
                            invalid_count += 1
                            logger.debug(
                                f"Invalid example at line {line_number} in {file_path}: "
                                f"missing required fields or empty values"
                            )
                    except json.JSONDecodeError as e:
                        invalid_count += 1
                        logger.warning(
                            f"Invalid JSON at line {line_number} in {file_path}: {e}"
                        )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Data file not found: {file_path}"
            ) from None
        except Exception as e:
            raise ValueError(
                f"Error reading data file {file_path}: {e}"
            ) from e
        
        if not examples:
            raise ValueError(
                f"No valid examples found in {file_path}. "
                f"Processed {line_number} lines, {invalid_count} invalid examples."
            )
        
        logger.info(
            f"Loaded {len(examples)} valid examples from {file_path} "
            f"({invalid_count} invalid examples skipped)"
        )
        
        return examples
        
    def _get_field_value(self, example: Any, field: str) -> Any:
        """Safely get a field value from an example (dict or Row object).
        
        Handles multiple access patterns:
        1. Dictionary access (for dict objects)
        2. Dictionary-style access with [] (for Row objects)
        3. Attribute access (for objects with attributes)
        
        Args:
            example: Example (dict, Row object, or other dict-like object)
            field: Field name to retrieve
            
        Returns:
            Field value or None if not found
            
        Raises:
            None: This function never raises exceptions, returns None on error
        """
        # Try dictionary access first (most common case)
        if isinstance(example, dict):
            return example.get(field)
        
        # Try dictionary-style access with [] operator (for Row objects)
        # This is the preferred method for HuggingFace Dataset Row objects
        try:
            if hasattr(example, "__getitem__"):
                return example[field]
        except (KeyError, TypeError, AttributeError):
            pass
        
        # Try attribute access as fallback (for custom objects)
        try:
            if hasattr(example, field):
                return getattr(example, field, None)
        except Exception:
            pass
        
        # Last resort: try to convert to dict and access
        try:
            if hasattr(example, "keys") and callable(getattr(example, "keys", None)):
                example_dict = {key: example[key] for key in example.keys()}
                return example_dict.get(field)
        except Exception:
            pass
        
        return None
    
    def _has_field(self, example: Any, field: str) -> bool:
        """Check if an example has a field.
        
        Handles multiple check patterns:
        1. Dictionary membership (for dict objects)
        2. Dictionary-style membership with 'in' operator (for Row objects)
        3. Attribute existence (for objects with attributes)
        
        Args:
            example: Example (dict, Row object, or other dict-like object)
            field: Field name to check
            
        Returns:
            True if field exists, False otherwise
        """
        # Try dictionary membership check first (most common case)
        if isinstance(example, dict):
            return field in example
        
        # Try dictionary-style membership with 'in' operator (for Row objects)
        # This is the preferred method for HuggingFace Dataset Row objects
        try:
            if hasattr(example, "__contains__"):
                return field in example
        except Exception:
            pass
        
        # Try keys() method (for dict-like objects)
        try:
            if hasattr(example, "keys") and callable(getattr(example, "keys", None)):
                return field in example.keys()
        except Exception:
            pass
        
        # Try attribute existence as fallback (for custom objects)
        try:
            return hasattr(example, field)
        except Exception:
            pass
        
        return False
    
    def _validate_example(self, example: Any) -> bool:
        """Validate that an example has the correct structure.
        
        Args:
            example: Example to validate (can be dict, Row object, or other)
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check required fields
        required_fields = ["input", "output"]
        for field in required_fields:
            # Check if field exists
            if not self._has_field(example, field):
                logger.debug(f"Missing required field: {field} in example of type {type(example)}")
                return False
            
            # Get field value
            value = self._get_field_value(example, field)
            if value is None:
                logger.debug(f"Field {field} has None value")
                return False
            
            # Ensure value is a non-empty string (or can be converted to one)
            try:
                value_str = str(value).strip()
                if not value_str:
                    logger.debug(f"Field {field} is empty after conversion to string")
                    return False
            except Exception as e:
                logger.debug(f"Cannot convert field {field} to string: {e}")
                return False
        
        return True
    
    def format_example(self, example: Any) -> Dict[str, str]:
        """Format an example using the chat template.
        
        Args:
            example: Input example with 'input' and 'output' keys, optionally 'system'.
                     Can be a dict or HuggingFace Dataset Row object.
            
        Returns:
            Formatted example with 'text' key
            
        Raises:
            ValueError: If tokenizer is not set or example format is invalid
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer is required for formatting examples")
        
        # Validate example first
        if not self._validate_example(example):
            # Try to get keys for better error message
            try:
                if isinstance(example, dict):
                    keys = list(example.keys())
                elif hasattr(example, "keys"):
                    keys = list(example.keys())
                else:
                    keys = "N/A"
            except Exception:
                keys = "N/A"
            
            logger.error(
                f"Invalid example format. Type: {type(example)}, "
                f"Keys: {keys}, Example: {example}"
            )
            raise ValueError(
                f"Invalid example format. Required fields: 'input', 'output'. "
                f"Found keys: {keys}"
            )
        
        # Extract fields using helper functions
        system_msg = self._get_field_value(example, "system")
        user_input = self._get_field_value(example, "input")
        assistant_output = self._get_field_value(example, "output")
        
        # Convert to strings and strip
        user_input = str(user_input).strip() if user_input else ""
        assistant_output = str(assistant_output).strip() if assistant_output else ""
        system_str = str(system_msg).strip() if system_msg else ""
        
        # Validate that we have non-empty input and output (validation should have caught this, but double-check)
        if not user_input:
            raise ValueError("Example 'input' field is empty or invalid")
        if not assistant_output:
            raise ValueError("Example 'output' field is empty or invalid")
        
        # Build messages list with optional system message
        messages = []
        
        # Add system message if present and non-empty
        if system_str:
            messages.append({"role": "system", "content": system_str})
        
        # Add user and assistant messages
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": assistant_output})
        
        # Apply chat template
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            raise ValueError(
                f"Failed to apply chat template to example: {e}. "
                f"Messages: {messages}"
            ) from e
        
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
        
        # Format the examples (process individually, not batched)
        train_dataset = datasets["train"].map(
            self.format_example,
            batched=False,
            remove_columns=datasets["train"].column_names,
            desc="Formatting training examples"
        )
        
        val_dataset = datasets["validation"].map(
            self.format_example,
            batched=False,
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
        # Process individually (batched=False) to handle Row objects properly
        train_dataset = datasets["train"].map(
            self.format_example,
            batched=False,
            remove_columns=datasets["train"].column_names,
            desc="Formatting training examples"
        )
        
        val_dataset = (
            datasets["validation"].map(
                self.format_example,
                batched=False,
                remove_columns=datasets["validation"].column_names,
                desc="Formatting validation examples"
            )
            if len(datasets["validation"]) > 0
            else None
        )
        
        # Shuffle if requested
        if shuffle:
            shuffle_seed = seed or self.config.shuffle_seed
            train_dataset = train_dataset.shuffle(seed=shuffle_seed)
            if val_dataset is not None:
                val_dataset = val_dataset.shuffle(seed=shuffle_seed)
        
        return train_dataset, val_dataset
