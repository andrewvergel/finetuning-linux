"""Custom callbacks for training."""
import os
import json
import logging
from typing import Dict, List, Optional, Any, Union

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)

class LoggingCallback(TrainerCallback):
    """A custom callback for logging training metrics to TensorBoard."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the logging callback.
        
        Args:
            log_dir: Directory to save TensorBoard logs
        """
        self.log_dir = log_dir
        self.writer = None
        self.global_step = 0
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Initialize TensorBoard writer at the start of training."""
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        logger.info(f"TensorBoard logging to {self.log_dir}")
        
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return
            
        if state.is_world_process_zero:
            if logs is not None:
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f"train/{key}", value, state.global_step)
                    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Close the TensorBoard writer at the end of training."""
        if self.writer is not None:
            self.writer.close()
            

class SavePeftModelCallback(TrainerCallback):
    """Callback to save the PEFT model during training."""
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Save the PEFT model and configuration."""
        if state.is_world_process_zero:
            checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            
            # Get the model from the trainer
            model = kwargs.get("model")
            if hasattr(model, "module"):
                model = model.module
                
            # Save the PEFT model
            model.save_pretrained(checkpoint_folder)
            
            # Save training arguments
            with open(os.path.join(checkpoint_folder, "training_args.json"), "w") as f:
                json.dump(args.to_dict(), f, indent=2)
                
            logger.info(f"Saved model checkpoint to {checkpoint_folder}")
            
        return control


class EarlyStoppingCallback(TrainerCallback):
    """A custom callback for early stopping during training."""
    
    def __init__(
        self,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
        metric_name: str = "eval_loss",
    ):
        """Initialize the early stopping callback.
        
        Args:
            early_stopping_patience: Number of evaluations to wait before stopping
            early_stopping_threshold: Minimum change in the monitored metric to qualify as improvement
            metric_name: Name of the metric to monitor
        """
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.metric_name = metric_name
        self.best_metric = None
        self.patience_counter = 0
        
    def check_metric_value(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Check if the metric has improved."""
        if not self.metric_name:
            return False
            
        metric_value = kwargs.get("metrics", {}).get(self.metric_name)
        if metric_value is None:
            logger.warning(f"Metric {self.metric_name} not found in metrics")
            return False
            
        if self.best_metric is None:
            self.best_metric = metric_value
            return True
            
        # Check if the metric has improved
        if metric_value < self.best_metric - self.early_stopping_threshold:
            self.best_metric = metric_value
            self.patience_counter = 0
            return True
            
        self.patience_counter += 1
        if self.patience_counter >= self.early_stopping_patience:
            logger.info(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
            control.should_training_stop = True
            
        return False
        
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Check if training should stop based on the evaluation metric."""
        if not self.check_metric_value(args, state, control, **kwargs):
            return
            
        # Reset patience counter if metric improved
        self.patience_counter = 0
        
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Log the best metric at the end of training."""
        if self.best_metric is not None:
            logger.info(f"Best {self.metric_name}: {self.best_metric:.4f}")
