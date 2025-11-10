"""Model building utilities for LoRA fine-tuning."""
import logging
import os
import torch
from typing import Dict, Optional, Tuple, Union, Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.peft_model import PeftModel

from finetuning_lora.config.model import ModelConfig
from finetuning_lora.config.training import TrainingConfig
from finetuning_lora.utils.logging import log_version_info

logger = logging.getLogger(__name__)

class ModelBuilder:
    """Handles model loading and preparation for LoRA fine-tuning."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: Optional[TrainingConfig] = None,
    ):
        """Initialize the model builder.
        
        Args:
            model_config: Model configuration
            training_config: Optional training configuration
        """
        self.model_config = model_config
        self.training_config = training_config or TrainingConfig()
        self.device = self._get_device()
        self._validate_configs()
        
    def _validate_configs(self):
        """Validate model and training configurations."""
        if self.model_config.load_in_4bit and self.model_config.load_in_8bit:
            raise ValueError("Cannot use both 4-bit and 8-bit quantization simultaneously")
            
        if self.model_config.load_in_4bit and not self.model_config.use_lora:
            logger.warning("4-bit quantization is most effective with LoRA. Consider enabling LoRA.")
            
    @staticmethod
    def _get_device() -> torch.device:
        """Get the appropriate device for training.
        
        Returns:
            torch.device: The device to use for training
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Enable TF32 for faster training on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            device = torch.device("cpu")
            logger.warning("No GPU found. Training on CPU will be very slow.")
            
        return device
        
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get the quantization configuration.
        
        Returns:
            Optional[BitsAndBytesConfig]: Quantization config if enabled, else None
        """
        if not (self.model_config.load_in_4bit or self.model_config.load_in_8bit):
            return None
            
        # Handle Pydantic Field objects in dataclass - get the actual value
        quant_type = self.model_config.bnb_4bit_quant_type
        compute_dtype = self.model_config.bnb_4bit_compute_dtype
        if hasattr(quant_type, 'default'):
            # It's a FieldInfo object, get the default value
            quant_type = quant_type.default
        if hasattr(compute_dtype, 'default'):
            # It's a FieldInfo object, get the default value
            compute_dtype = compute_dtype.default
        
        # Get string values from enums
        quant_type_str = quant_type.value if hasattr(quant_type, 'value') else quant_type
        compute_dtype_str = compute_dtype.value if hasattr(compute_dtype, 'value') else compute_dtype
        
        return BitsAndBytesConfig(
            load_in_4bit=self.model_config.load_in_4bit,
            load_in_8bit=self.model_config.load_in_8bit,
            bnb_4bit_quant_type=quant_type_str,
            bnb_4bit_compute_dtype=getattr(torch, compute_dtype_str),
            bnb_4bit_use_double_quant=self.model_config.use_nested_quant,
        )
        
    def _get_lora_config(self) -> Optional[LoraConfig]:
        """Get the LoRA configuration.
        
        Returns:
            Optional[LoraConfig]: LoRA config if enabled, else None
        """
        if not self.model_config.use_lora:
            return None
            
        return LoraConfig(
            r=self.model_config.lora_rank,
            lora_alpha=self.model_config.lora_alpha,
            target_modules=self.model_config.lora_target_modules,
            lora_dropout=self.model_config.lora_dropout,
            bias=self.model_config.lora_bias,
            task_type="CAUSAL_LM",
        )
        
    def load_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        """Load the tokenizer.
        
        Returns:
            Tokenizer instance
        """
        logger.info(f"Loading tokenizer from {self.model_config.model_name_or_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name_or_path,
            trust_remote_code=self.model_config.trust_remote_code,
            use_fast=self.model_config.use_fast_tokenizer,
            padding_side=self.model_config.padding_side,
            token=self.model_config.use_auth_token,
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                logger.warning("Added [PAD] token to tokenizer")
                
        return tokenizer
        
    def load_model(self) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
        """Load and prepare the model and tokenizer for training.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        log_version_info()
        logger.info(f"Loading model from {self.model_config.model_name_or_path}")
        
        # Load tokenizer
        tokenizer = self.load_tokenizer()
        
        # Get quantization config
        quantization_config = self._get_quantization_config()
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=self.model_config.trust_remote_code,
            use_auth_token=self.model_config.use_auth_token,
            torch_dtype="auto",
        )
        
        # Prepare for k-bit training if using quantization
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.training_config.gradient_checkpointing,
            )
            
        # Apply LoRA if enabled
        lora_config = self._get_lora_config()
        if lora_config is not None:
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
        # Configure gradient checkpointing if enabled
        if self.training_config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            
        # Set model to evaluation mode by default
        model.eval()
        
        return model, tokenizer
        
    def save_model(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        output_dir: str,
        save_tokenizer: bool = True,
    ) -> None:
        """Save the model and tokenizer.
        
        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            output_dir: Directory to save to
            save_tokenizer: Whether to save the tokenizer
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save PEFT model
        if isinstance(model, PeftModel):
            model.save_pretrained(output_dir)
            if save_tokenizer:
                tokenizer.save_pretrained(output_dir)
        else:
            # Save full model
            model.save_pretrained(output_dir)
            if save_tokenizer:
                tokenizer.save_pretrained(output_dir)
                
        logger.info(f"Model saved to {output_dir}")
