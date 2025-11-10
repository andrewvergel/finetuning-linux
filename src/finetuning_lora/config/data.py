"""Data configuration for the finetuning pipeline."""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class DataConfig:
    """Configuration for data processing.
    
    Attributes:
        train_path: Path to the training data file
        val_path: Path to the validation data file (optional)
        max_length: Maximum sequence length
        padding_side: Padding side ('left' or 'right')
        truncation: Whether to truncate sequences that exceed max_length
        padding: Padding strategy ('max_length' or 'longest')
        return_tensors: Return format for tokenizer ('pt' for PyTorch)
        validation_split: Fraction of training data to use for validation
        shuffle_seed: Random seed for shuffling data
    """
    train_path: str = "data/instructions.jsonl"
    val_path: Optional[str] = None
    max_length: int = 512
    padding_side: str = "right"
    truncation: bool = True
    padding: str = "max_length"
    return_tensors: str = "pt"
    validation_split: float = 0.1
    shuffle_seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {
            "train_path": self.train_path,
            "val_path": self.val_path,
            "max_length": self.max_length,
            "padding_side": self.padding_side,
            "truncation": self.truncation,
            "padding": self.padding,
            "return_tensors": self.return_tensors,
            "validation_split": self.validation_split,
            "shuffle_seed": self.shuffle_seed,
        }
    
    def get_tokenizer_kwargs(self) -> Dict[str, Any]:
        """Get tokenizer keyword arguments from config.
        
        Note: padding, truncation, and return_tensors are handled separately 
        in the processor with explicit values.
        
        Returns:
            Dictionary of tokenizer arguments
        """
        return {
            "max_length": self.max_length,
        }
