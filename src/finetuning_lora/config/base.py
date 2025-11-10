"""Base configuration for the finetuning project."""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os
import torch

@dataclass
class BaseConfig:
    """Base configuration class with common settings."""
    # General settings
    project_name: str = "finetuning-lora"
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    log_level: str = "INFO"
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True
    
    # Training settings
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # LoRA settings
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    
    # QLoRA settings (if enabled)
    use_qlora: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    use_nested_quant: bool = True
    
    # Output settings
    output_dir: str = "models/out-lora"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    
    # Data settings
    train_file: str = "data/instructions.jsonl"
    validation_split: float = 0.1
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        env_vars = {k: v for k, v in os.environ.items() if k.startswith("FT_")}
        config_dict = {}
        
        for key, value in env_vars.items():
            # Convert FT_MODEL_NAME to model_name, etc.
            config_key = key[3:].lower()
            
            # Convert string values to appropriate types
            if value.lower() in ('true', 'false'):
                config_dict[config_key] = value.lower() == 'true'
            elif value.isdigit():
                config_dict[config_key] = int(value)
            else:
                try:
                    config_dict[config_key] = float(value)
                except ValueError:
                    config_dict[config_key] = value
        
        self.update_from_dict(config_dict)
