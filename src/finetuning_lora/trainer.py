"""Training loop and utilities for the finetuning project."""
import os
import logging
import json
import torch
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import PeftModel

from .config.training import TrainingConfig
from .models.model import ModelLoader
from .data.dataset import DataProcessor

logger = logging.getLogger(__name__)

class SavePeftModelCallback(TrainerCallback):
    """Callback to save the PEFT model and adapter configuration."""
    
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Save the PEFT model and adapter configuration."""
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        
        # Save adapter model
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        
        # Save training arguments
        training_args_path = os.path.join(checkpoint_folder, "training_args.bin")
        torch.save(args, training_args_path)
        
        return control

class FineTuningTrainer:
    """Handles the fine-tuning process with LoRA."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize with training configuration."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def train(self):
        """Run the training process."""
        # Setup logging
        self._setup_logging()
        
        # Load model and tokenizer
        model_loader = ModelLoader(self.config)
        self.model, self.tokenizer = model_loader.load_model_and_tokenizer()
        
        # Prepare data
        data_processor = DataProcessor(
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length
        )
        
        datasets = data_processor.prepare_for_training(
            train_file=self.config.train_file,
            test_size=0.1  # Use 10% of training data for validation if no validation set
        )
        
        # Initialize data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Prepare training arguments
        training_args = self._prepare_training_arguments()
        
        # Initialize Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation"),
            data_collator=data_collator,
            callbacks=[SavePeftModelCallback],
        )
        
        # Train the model
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save the final model
        self._save_model()
        
        # Log training metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(datasets["train"])
        
        if "validation" in datasets:
            metrics.update(self.trainer.evaluate())
        
        # Save metrics
        metrics_file = os.path.join(self.config.output_dir, "training_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training completed. Metrics saved to {metrics_file}")
        return metrics
    
    def _setup_logging(self):
        """Configure logging."""
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.config.output_dir, "training.log")),
            ],
        )
        
        # Set log level for datasets and other libraries
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
    
    def _prepare_training_arguments(self) -> TrainingArguments:
        """Prepare training arguments from config."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            num_train_epochs=self.config.num_train_epochs,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=self.config.max_grad_norm,
            report_to=["tensorboard"],
            remove_unused_columns=False,
            logging_first_step=True,
            save_safetensors=True,
            dataloader_num_workers=self.config.preprocessing_num_workers,
            dataloader_pin_memory=True,
        )
    
    def _save_model(self):
        """Save the final model and tokenizer."""
        output_dir = os.path.join(self.config.output_dir, "final_model")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save adapter model
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save model card
        self._save_model_card(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
    
    def _save_model_card(self, output_dir: str):
        """Save a model card with training information."""
        model_card = f"""---
language: es
license: apache-2.0
base_model: {self.config.model_name_or_path}
---

# Model Card for {os.path.basename(output_dir)}

## Model Details

- **Model type:** Causal Language Model with LoRA
- **Language(s) (NLP):** Spanish
- **Finetuned from model:** {self.config.model_name_or_path}
- **License:** Apache 2.0

## Training Details

### Training Data

- Training examples: [Specify number]
- Evaluation examples: [Specify number]

### Training Hyperparameters

- Learning Rate: {self.config.learning_rate}
- Batch Size: {self.config.per_device_train_batch_size}
- Epochs: {self.config.num_train_epochs}
- Weight Decay: {self.config.weight_decay}

## Intended Use

This model is intended to be used for [specify use case].

## Limitations and Bias

[More information needed]

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

model_name = "{output_dir}"
config = PeftConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, model_name)
model.eval()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate text
def generate(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```
"""
        with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card)
