"""Model configuration for the finetuning project.

This module provides configuration for loading and adapting language models with LoRA.
It includes settings for model loading, quantization, LoRA adaptation, and text generation.

Example:
    ```python
    config = ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit=True,
        lora_rank=8,
        lora_alpha=16
    )
    peft_config = config.to_peft_config()
    ```
"""
from enum import Enum
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator, ConfigDict


class LoraBiasType(str, Enum):
    """Bias type for LoRA layers.
    
    - NONE: No bias
    - ALL: Train bias for all layers
    - LORA_ONLY: Only train bias for LoRA layers
    """
    NONE = "none"
    ALL = "all"
    LORA_ONLY = "lora_only"


class QuantizationType(str, Enum):
    """Supported quantization types."""
    NF4 = "nf4"
    FP4 = "fp4"
    INT8 = "int8"


class ComputeDtype(str, Enum):
    """Supported compute dtypes for quantization."""
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"

class ModelConfig(BaseModel):
    """Configuration for model loading and LoRA adaptation.
    
    Attributes:
        model_name_or_path: Path or name of the pre-trained model.
        trust_remote_code: Whether to trust remote code when loading the model.
        use_auth_token: Whether to use authentication token for private models.
        load_in_4bit: Whether to load the model in 4-bit precision.
        load_in_8bit: Whether to load the model in 8-bit precision.
        bnb_4bit_quant_type: Quantization type for 4-bit quantization.
        bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization.
        use_nested_quant: Whether to use nested quantization.
        use_lora: Whether to use LoRA adaptation.
        lora_rank: Rank of LoRA update matrices.
        lora_alpha: Alpha parameter for LoRA scaling.
        lora_dropout: Dropout probability for LoRA layers.
        lora_target_modules: List of module names to apply LoRA to.
        lora_bias: Type of bias to use in LoRA layers.
    """
    
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )
    # Model settings
    model_name_or_path: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="Path or name of the pre-trained model"
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Whether to trust remote code when loading the model"
    )
    use_auth_token: bool = Field(
        default=False,
        description="Whether to use authentication token for private models"
    )
    
    # Quantization settings (for QLoRA)
    load_in_4bit: bool = Field(
        default=True,
        description="Whether to load the model in 4-bit precision"
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Whether to load the model in 8-bit precision"
    )
    bnb_4bit_quant_type: QuantizationType = Field(
        default=QuantizationType.NF4,
        description="Quantization type for 4-bit quantization"
    )
    bnb_4bit_compute_dtype: ComputeDtype = Field(
        default=ComputeDtype.BFLOAT16,
        description="Compute dtype for 4-bit quantization"
    )
    use_nested_quant: bool = Field(
        default=True,
        description="Whether to use nested quantization"
    )
    
    # LoRA settings
    use_lora: bool = Field(
        default=True,
        description="Whether to use LoRA adaptation"
    )
    lora_rank: int = Field(
        default=8,
        ge=1,
        le=128,
        description="Rank of LoRA update matrices"
    )
    lora_alpha: int = Field(
        default=16,
        ge=1,
        description="Alpha parameter for LoRA scaling"
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Dropout probability for LoRA layers"
    )
    lora_target_modules: List[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"],
        description="List of module names to apply LoRA to"
    )
    lora_bias: LoraBiasType = Field(
        default=LoraBiasType.NONE,
        description="Type of bias to use in LoRA layers"
    )
    
    # Tokenizer settings
    use_fast_tokenizer: bool = Field(
        default=True,
        description="Whether to use the fast tokenizer"
    )
    padding_side: Literal["left", "right"] = Field(
        default="right",
        description="Side to pad to for tokenization"
    )
    
    # Generation settings
    max_new_tokens: int = Field(
        default=128,
        ge=1,
        le=4096,
        description="Maximum number of new tokens to generate"
    )
    temperature: float = Field(
        default=0.7,
        gt=0.0,
        le=2.0,
        description="Value used to modulate the next token probabilities"
    )
    top_p: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling: only the smallest set of most probable tokens with probabilities that add up to top_p"
    )
    top_k: int = Field(
        default=50,
        ge=1,
        description="Number of highest probability vocabulary tokens to keep for top-k filtering"
    )
    repetition_penalty: float = Field(
        default=1.1,
        ge=1.0,
        le=2.0,
        description="Penalty for repeated tokens"
    )
    do_sample: bool = Field(
        default=True,
        description="Whether to use sampling; use greedy decoding otherwise"
    )
    
    @model_validator(mode='after')
    def validate_quantization_mutually_exclusive(self):
        """Ensure 4-bit and 8-bit quantization are mutually exclusive."""
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot use both 4-bit and 8-bit quantization simultaneously")
        return self
    
    def to_peft_config(self) -> Dict[str, Any]:
        """Convert to PEFT (LoRA) configuration.
        
        Returns:
            Dict containing the LoRA configuration. Returns an empty dict if use_lora is False.
            
        Example:
            ```python
            config = ModelConfig()
            peft_config = config.to_peft_config()
            ```
        """
        if not self.use_lora:
            return {}
            
        return {
            "r": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.lora_target_modules,
            "bias": self.lora_bias,  # Already a string due to use_enum_values=True
            "task_type": "CAUSAL_LM",
        }
    
    def to_quantization_config(self) -> Dict[str, Any]:
        """Convert to quantization configuration.
        
        Returns:
            Dict containing the quantization configuration. Returns an empty dict
            if neither 4-bit nor 8-bit quantization is enabled.
            
        Raises:
            ValueError: If both 4-bit and 8-bit quantization are enabled.
        """
        if not (self.load_in_4bit or self.load_in_8bit):
            return {}
            
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot use both 4-bit and 8-bit quantization simultaneously")
            
        config = {
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "bnb_4bit_use_double_quant": self.use_nested_quant,
        }
        
        if self.load_in_4bit:
            config.update({
                "bnb_4bit_quant_type": self.bnb_4bit_quant_type,  # Already a string due to use_enum_values=True
                "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,  # Already a string due to use_enum_values=True
            })
            
        return config
    
    def to_generation_config(self) -> Dict[str, Any]:
        """Convert to generation configuration.
        
        Returns:
            Dict containing the generation configuration.
            
        Note:
            The `pad_token_id` and `eos_token_id` should be set using the actual
            tokenizer's values in the training/inference script.
        """
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "pad_token_id": 0,  # Will be overridden by tokenizer.pad_token_id
            "eos_token_id": 1,  # Will be overridden by tokenizer.eos_token_id
        }
