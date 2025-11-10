"""Training utilities for LoRA fine-tuning."""
import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from finetuning_lora.config import ModelConfig, TrainingConfig, DataConfig
from finetuning_lora.training.callbacks import (
    LoggingCallback,
    SavePeftModelCallback,
    EarlyStoppingCallback,
)
from finetuning_lora.utils.logging import setup_logging, log_version_info

logger = logging.getLogger(__name__)

class LoRATrainer:
    """Handles the training process for LoRA fine-tuning."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
    ):
        """Initialize the trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            data_config: Data configuration
        """
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
        # Set up logging
        setup_logging()
        log_version_info()
        
        # Set seed for reproducibility
        set_seed(self.training_config.seed)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def _get_training_arguments(self) -> TrainingArguments:
        """Create training arguments from config.
        
        Returns:
            TrainingArguments instance
        """
        return TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.per_device_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            lr_scheduler_type=self.training_config.lr_scheduler,
            save_strategy=self.training_config.save_strategy,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            evaluation_strategy=self.training_config.eval_strategy,
            eval_steps=self.training_config.eval_steps,
            logging_steps=self.training_config.logging_steps,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            seed=self.training_config.seed,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            remove_unused_columns=self.training_config.remove_unused_columns,
            report_to=self.training_config.report_to,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
        )
        
    def _get_callbacks(self) -> List[object]:
        """Get training callbacks.
        
        Returns:
            List of callbacks
        """
        callbacks = [
            LoggingCallback(log_dir=os.path.join(self.training_config.output_dir, "logs")),
            SavePeftModelCallback(),
        ]
        
        if self.training_config.early_stopping:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience,
                    early_stopping_threshold=self.training_config.early_stopping_threshold,
                    metric_name="eval_loss",
                )
            )
            
        return callbacks
    
    def _get_data_collator(self):
        """Get data collator for language modeling."""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
    
    def train(
        self,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
    ) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
        """Run the training loop.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            
        Returns:
            Tuple of (trained model, tokenizer)
        """
        # Set up training arguments
        training_args = self._get_training_arguments()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.training_config.output_dir, exist_ok=True)
        
        # Initialize model and tokenizer if not provided
        if self.model is None or self.tokenizer is None:
            from finetuning_lora.models.builder import ModelBuilder
            
            model_builder = ModelBuilder(
                model_config=self.model_config,
                training_config=self.training_config,
            )
            self.model, self.tokenizer = model_builder.load_model()
        
        # Get data collator and callbacks
        data_collator = self._get_data_collator()
        callbacks = self._get_callbacks()
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        # Train the model
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save the final model
        self.trainer.save_model(self.training_config.output_dir)
        self.tokenizer.save_pretrained(self.training_config.output_dir)
        
        # Log training results
        logger.info(f"Training completed. Results saved to {self.training_config.output_dir}")
        logger.info(f"Training metrics: {train_result.metrics}")
        
        return self.model, self.tokenizer
    
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """Evaluate the model on the evaluation dataset.
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer has not been initialized. Call train() first.")
            
        logger.info("Running evaluation...")
        metrics = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        # Log metrics
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def save_model(self, output_dir: str):
        """Save the model and tokenizer.
        
        Args:
            output_dir: Directory to save the model and tokenizer
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be initialized before saving.")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)
            
        # Save the tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model and tokenizer saved to {output_dir}")
