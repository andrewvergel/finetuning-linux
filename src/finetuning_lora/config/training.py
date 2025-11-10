"""Training configuration for the finetuning project."""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from .base import BaseConfig

@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for model training."""
    # Training settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    
    # Evaluation settings
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    
    # Mixed precision training
    fp16: bool = False
    bf16: bool = True
    
    # Gradient checkpointing
    gradient_checkpointing: bool = True
    
    # Data processing
    max_seq_length: int = 1024
    preprocessing_num_workers: Optional[int] = None
    
    # Packing (for more efficient training)
    packing: bool = False
    
    # SFTTrainer specific parameters
    neftune_noise_alpha: Optional[float] = None
    dataset_num_proc: Optional[int] = None
    dataset_text_field: str = "text"
    
    # Early stopping configuration
    early_stopping: bool = True
    
    # Reporting
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # DataLoader settings
    # Note: num_workers=0 avoids multiprocessing serialization issues with SFTTrainer
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    
    # Save strategy
    save_strategy: str = "steps"
    
    def to_training_args(self):
        """Convert to TrainingArguments compatible dictionary."""
        return {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "num_train_epochs": self.num_train_epochs,
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "max_grad_norm": self.max_grad_norm,
            "report_to": self.report_to,
            "remove_unused_columns": self.remove_unused_columns,
            "logging_first_step": True,
            "save_safetensors": True,
            # Ensure dataloader_num_workers is always an integer (not None)
            # Since dataloader_num_workers is int = 0 (not Optional), it should always be an int
            # But handle edge case where it might be None (e.g., from env loading)
            "dataloader_num_workers": (
                self.dataloader_num_workers
                if isinstance(self.dataloader_num_workers, int)
                else (
                    self.preprocessing_num_workers
                    if isinstance(self.preprocessing_num_workers, int)
                    else 0
                )
            ),
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "tf32": True,  # Enable TF32 for faster training on Ampere GPUs
        }
