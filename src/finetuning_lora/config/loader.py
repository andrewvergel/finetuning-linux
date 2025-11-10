"""Configuration loader utilities for loading configs from environment variables."""
import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .model import ModelConfig
from .training import TrainingConfig
from .data import DataConfig

logger = logging.getLogger(__name__)


def load_env_file(path: str = ".env") -> None:
    """Load environment variables from a .env file.
    
    Args:
        path: Path to the .env file
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    
    try:
        with open(env_path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    except Exception as exc:
        logger.warning(f"Could not load {path}: {exc}")


def _parse_env_value(value: str) -> Any:
    """Parse an environment variable value to the appropriate type.
    
    Args:
        value: String value from environment variable
        
    Returns:
        Parsed value with appropriate type
    """
    if not value:
        return None
    
    # Handle boolean values
    if value.lower() in ("true", "1", "yes", "y", "on"):
        return True
    if value.lower() in ("false", "0", "no", "n", "off"):
        return False
    
    # Handle list values (comma-separated)
    if "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]
    
    # Handle integer values
    try:
        if "." not in value:  # Avoid parsing floats as ints
            return int(value)
    except ValueError:
        pass
    
    # Handle float values
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string
    return value


def _get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables.
    
    Returns:
        Dictionary of configuration values with FT_ prefix removed
    """
    config = {}
    for key, value in os.environ.items():
        if key.startswith("FT_"):
            # Remove FT_ prefix and convert to lowercase
            config_key = key[3:].lower()
            config[config_key] = _parse_env_value(value)
    return config


def _apply_config_to_dataclass(config_dict: Dict[str, Any], config_obj: Any) -> Any:
    """Apply configuration dictionary to a dataclass instance.
    
    Args:
        config_dict: Dictionary of configuration values
        config_obj: Dataclass instance to update
        
    Returns:
        Updated config object
    """
    for key, value in config_dict.items():
        if hasattr(config_obj, key):
            setattr(config_obj, key, value)
        else:
            # Try with underscore variations
            key_underscore = key.replace("-", "_")
            if hasattr(config_obj, key_underscore):
                setattr(config_obj, key_underscore, value)
    return config_obj


def load_model_config(env_file: Optional[str] = ".env") -> ModelConfig:
    """Load ModelConfig from environment variables.
    
    Args:
        env_file: Path to .env file (optional)
        
    Returns:
        ModelConfig instance
    """
    if env_file:
        load_env_file(env_file)
    
    env_config = _get_env_config()
    
    # Map environment variables to ModelConfig fields
    model_config_dict = {}
    
    # Model settings
    if "model_id" in env_config:
        model_config_dict["model_name_or_path"] = env_config["model_id"]
    if "trust_remote_code" in env_config:
        model_config_dict["trust_remote_code"] = env_config["trust_remote_code"]
    
    # Quantization settings
    if "use_qlora" in env_config:
        model_config_dict["load_in_4bit"] = env_config["use_qlora"]
    if "load_in_4bit" in env_config:
        model_config_dict["load_in_4bit"] = env_config["load_in_4bit"]
    if "load_in_8bit" in env_config:
        model_config_dict["load_in_8bit"] = env_config["load_in_8bit"]
    
    # LoRA settings
    if "lora_rank" in env_config:
        model_config_dict["lora_rank"] = env_config["lora_rank"]
    if "lora_alpha" in env_config:
        model_config_dict["lora_alpha"] = env_config["lora_alpha"]
    if "lora_dropout" in env_config:
        model_config_dict["lora_dropout"] = env_config["lora_dropout"]
    if "lora_target_modules" in env_config:
        target_modules = env_config["lora_target_modules"]
        if isinstance(target_modules, str):
            # Handle comma-separated string
            if "," in target_modules:
                model_config_dict["lora_target_modules"] = [m.strip() for m in target_modules.split(",")]
            else:
                model_config_dict["lora_target_modules"] = [target_modules]
        elif isinstance(target_modules, list):
            model_config_dict["lora_target_modules"] = target_modules
    if "use_lora" in env_config:
        model_config_dict["use_lora"] = env_config["use_lora"]
    
    # Create config with defaults and update from env
    config = ModelConfig(**model_config_dict)
    
    return config


def load_training_config(env_file: Optional[str] = ".env") -> TrainingConfig:
    """Load TrainingConfig from environment variables.
    
    Args:
        env_file: Path to .env file (optional)
        
    Returns:
        TrainingConfig instance
    """
    if env_file:
        load_env_file(env_file)
    
    env_config = _get_env_config()
    
    # Map environment variables to TrainingConfig fields
    training_config_dict = {}
    
    # Training settings
    if "num_epochs" in env_config:
        training_config_dict["num_train_epochs"] = env_config["num_epochs"]
    if "per_device_batch_size" in env_config:
        training_config_dict["per_device_train_batch_size"] = env_config["per_device_batch_size"]
        training_config_dict["per_device_eval_batch_size"] = env_config["per_device_batch_size"]
    if "per_device_train_batch_size" in env_config:
        training_config_dict["per_device_train_batch_size"] = env_config["per_device_train_batch_size"]
    if "gradient_accumulation" in env_config:
        training_config_dict["gradient_accumulation_steps"] = env_config["gradient_accumulation"]
    if "learning_rate" in env_config:
        training_config_dict["learning_rate"] = env_config["learning_rate"]
    if "weight_decay" in env_config:
        training_config_dict["weight_decay"] = env_config["weight_decay"]
    if "warmup_ratio" in env_config:
        training_config_dict["warmup_ratio"] = env_config["warmup_ratio"]
    if "lr_scheduler" in env_config:
        training_config_dict["lr_scheduler_type"] = env_config["lr_scheduler"]
    
    # Evaluation and logging
    if "eval_steps" in env_config:
        training_config_dict["eval_steps"] = env_config["eval_steps"]
    if "save_steps" in env_config:
        training_config_dict["save_steps"] = env_config["save_steps"]
    if "save_total_limit" in env_config:
        training_config_dict["save_total_limit"] = env_config["save_total_limit"]
    if "logging_steps" in env_config:
        training_config_dict["logging_steps"] = env_config["logging_steps"]
    if "save_strategy" in env_config:
        training_config_dict["save_strategy"] = env_config["save_strategy"]
    
    # Memory and optimization
    if "max_seq_len" in env_config:
        training_config_dict["max_seq_length"] = env_config["max_seq_len"]
    if "force_packing" in env_config:
        training_config_dict["packing"] = env_config["force_packing"]
    if "packing" in env_config:
        training_config_dict["packing"] = env_config["packing"]
    
    # SFTTrainer specific
    if "neftune_noise_alpha" in env_config:
        training_config_dict["neftune_noise_alpha"] = env_config["neftune_noise_alpha"]
    if "dataset_num_proc" in env_config:
        training_config_dict["dataset_num_proc"] = env_config["dataset_num_proc"]
    
    # Mixed precision
    if "bf16" in env_config:
        training_config_dict["bf16"] = env_config["bf16"]
    if "fp16" in env_config:
        training_config_dict["fp16"] = env_config["fp16"]
    
    # Gradient checkpointing
    if "gradient_checkpointing" in env_config:
        training_config_dict["gradient_checkpointing"] = env_config["gradient_checkpointing"]
    
    # Output directory
    if "out_dir" in env_config:
        training_config_dict["output_dir"] = env_config["out_dir"]
    
    # Early stopping
    if "early_stopping_patience" in env_config:
        training_config_dict["early_stopping_patience"] = env_config["early_stopping_patience"]
    if "early_stopping_threshold" in env_config:
        training_config_dict["early_stopping_threshold"] = env_config["early_stopping_threshold"]
    
    # Create config with defaults and update from env
    config = TrainingConfig()
    _apply_config_to_dataclass(training_config_dict, config)
    
    return config


def load_data_config(env_file: Optional[str] = ".env") -> DataConfig:
    """Load DataConfig from environment variables.
    
    Args:
        env_file: Path to .env file (optional)
        
    Returns:
        DataConfig instance
    """
    if env_file:
        load_env_file(env_file)
    
    env_config = _get_env_config()
    
    # Map environment variables to DataConfig fields
    data_config_dict = {}
    
    if "data_path" in env_config:
        data_config_dict["train_path"] = env_config["data_path"]
    if "train_path" in env_config:
        data_config_dict["train_path"] = env_config["train_path"]
    if "val_path" in env_config:
        data_config_dict["val_path"] = env_config["val_path"]
    if "validation_split" in env_config:
        data_config_dict["validation_split"] = env_config["validation_split"]
    if "dataset_shuffle_seed" in env_config:
        data_config_dict["shuffle_seed"] = env_config["dataset_shuffle_seed"]
    if "max_seq_len" in env_config:
        data_config_dict["max_length"] = env_config["max_seq_len"]
    
    # Create config with defaults and update from env
    config = DataConfig()
    _apply_config_to_dataclass(data_config_dict, config)
    
    return config

