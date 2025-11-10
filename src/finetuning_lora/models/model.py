"""Model loading and setup for the finetuning project.

This module provides components for loading and configuring language models with support for
quantization and LoRA fine-tuning. It follows the Single Responsibility Principle by separating
concerns into specialized classes.

Example:
    ```python
    # Basic usage
    config = ModelConfig(
        model_name_or_path="gpt2",
        use_lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    
    loader = create_model_loader(config)
    model, tokenizer = loader.load_model_and_tokenizer()
    ```
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Optional,
    Tuple,
    Type,
    TypeVar,
    final,
    TypedDict,
    Union,
)

import torch
from torch import cuda, nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from ..config.model import ModelConfig
from ..exceptions import ModelLoadingError, TokenizerLoadingError

# Type aliases
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
ModelType = TypeVar('ModelType', bound=PreTrainedModel)

class LoraConfigDict(TypedDict, total=False):
    """Type definition for LoRA configuration dictionary."""
    r: int
    lora_alpha: float
    target_modules: list[str]
    lora_dropout: float
    bias: str

logger = logging.getLogger(__name__)

# Type variable for generic model types
T = TypeVar('T', bound=PreTrainedModel)

class ModelInitializationError(ModelLoadingError):
    """Raised when model initialization fails."""
    pass

class TokenizerInitializationError(TokenizerLoadingError):
    """Raised when tokenizer initialization fails."""
    pass

@final
class TokenizerLoader:
    """Handles loading and configuration of tokenizers.
    
    This class is responsible for loading and configuring tokenizers with proper
    error handling and validation.
    
    Attributes:
        DEFAULT_PADDING_SIDE: Default padding side for tokenizers.
    """
    
    DEFAULT_PADDING_SIDE: ClassVar[str] = 'right'
    
    @classmethod
    def load(cls, config: ModelConfig) -> TokenizerType:
        """Load and configure a tokenizer from the given configuration.
        
        Args:
            config: Model configuration containing tokenizer settings.
                Must include model_name_or_path and other tokenizer parameters.
                
        Returns:
            TokenizerType: Configured tokenizer instance.
            
        Raises:
            TokenizerInitializationError: If tokenizer loading or configuration fails.
            
        Example:
            ```python
            config = ModelConfig(
                model_name_or_path="gpt2",
                use_fast_tokenizer=True,
                padding_side='right'
            )
            tokenizer = TokenizerLoader.load(config)
            ```
        """
        if not config.model_name_or_path:
            raise TokenizerInitializationError("Model name or path is required")
            
        padding_side = config.padding_side or cls.DEFAULT_PADDING_SIDE
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name_or_path,
                trust_remote_code=config.trust_remote_code,
                use_fast=config.use_fast_tokenizer,
                padding_side=config.padding_side,
                use_auth_token=config.use_auth_token,
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            return tokenizer
            
        except Exception as e:
            raise TokenizerLoadingError(
                f"Failed to load tokenizer from {config.model_name_or_path}"
            ) from e

@final
class QuantizationConfigFactory:
    """Factory for creating quantization configurations.
    
    This class handles the creation of BitsAndBytesConfig objects with proper
    validation of input parameters.
    
    Attributes:
        COMPUTE_DTYPE_MAP: Mapping of string dtype names to torch dtypes.
        SUPPORTED_QUANT_TYPES: Set of supported quantization types.
    """
    
    COMPUTE_DTYPE_MAP: ClassVar[Dict[str, torch.dtype]] = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    
    SUPPORTED_QUANT_TYPES: ClassVar[set[str]] = {'nf4', 'fp4'}
    
    @classmethod
    def create(cls, config: ModelConfig) -> Optional[BitsAndBytesConfig]:
        """Create a quantization configuration based on the model config.
        
        Args:
            config: Model configuration with quantization settings
            
        Returns:
            Configured BitsAndBytesConfig or None if quantization is disabled
        """
        if not (config.load_in_4bit or config.load_in_8bit):
            return None

        return BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=cls.COMPUTE_DTYPE_MAP.get(
                config.bnb_4bit_compute_dtype, torch.bfloat16
            ),
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )

@final
class LoraConfigurator:
    """Handles LoRA configuration and model preparation.
    
    This class is responsible for configuring and applying LoRA to a pre-trained
    model, including k-bit training preparation if needed.
    """
    
    @classmethod
    def prepare(
        cls,
        model: PreTrainedModel,
        config: ModelConfig
    ) -> PreTrainedModel:
        """Prepare a model for LoRA fine-tuning.
        
        Args:
            model: The pre-trained model to apply LoRA to.
            config: Configuration containing LoRA parameters.
                Must include lora_rank, lora_alpha, lora_dropout, etc.
                
        Returns:
            PreTrainedModel: The model configured for LoRA training.
            
        Raises:
            ModelInitializationError: If LoRA configuration or application fails.
            
        Example:
            ```python
            config = ModelConfig(
                lora_rank=8,
                lora_alpha=32,
                lora_dropout=0.1,
                lora_target_modules=['q_proj', 'v_proj']
            )
            model = LoraConfigurator.prepare(model, config)
            ```
        """
        try:
            # Prepare for k-bit training if needed
            if config.load_in_4bit or config.load_in_8bit:
                model = prepare_model_for_kbit_training(model)
            
            # Create and apply LoRA config
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias=config.lora_bias,
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            return model
            
        except Exception as e:
            raise ModelLoadingError("Failed to prepare LoRA model") from e

@final
class ModelLoader(Generic[ModelType]):
    """Handles loading and preparing models for training and inference.
    
    This class follows the Single Responsibility Principle by delegating specific
    tasks to specialized components (TokenizerLoader, QuantizationConfigFactory, etc.).
    It provides a clean interface for loading models with various configurations
    including quantization and LoRA fine-tuning.
    
    Attributes:
        config: The model configuration.
        tokenizer_loader: Instance for loading tokenizers.
        quant_config_factory: Factory for creating quantization configs.
        lora_configurator: Configurator for LoRA fine-tuning.
    """
    
    __slots__ = ['config', 'tokenizer_loader', 'quant_config_factory', 'lora_configurator']
    
    def __init__(
        self,
        config: ModelConfig,
        tokenizer_loader: Optional[TokenizerLoader] = None,
        quant_config_factory: Optional[Type[QuantizationConfigFactory]] = None,
        lora_configurator: Optional[LoraConfigurator] = None,
    ) -> None:
        """Initialize the model loader with dependencies.
        
        Args:
            config: Model configuration containing all necessary parameters.
            tokenizer_loader: Optional tokenizer loader instance. If None, a default
                TokenizerLoader will be used.
            quant_config_factory: Optional quantization config factory class. If None,
                the default QuantizationConfigFactory will be used.
            lora_configurator: Optional LoRA configurator instance. If None, a default
                LoraConfigurator will be used.
                
        Raises:
            ValueError: If the provided config is None.
        """
        if config is None:
            raise ValueError("Config cannot be None")
        self.config = config
        self.tokenizer_loader = tokenizer_loader or TokenizerLoader()
        self.quant_config_factory = quant_config_factory or QuantizationConfigFactory
        self.lora_configurator = lora_configurator or LoraConfigurator()
    
    def load_model_and_tokenizer(self) -> Tuple[T, PreTrainedTokenizer]:
        """Load model and tokenizer based on configuration.
        
        Returns:
            Tuple of (model, tokenizer) ready for training or inference
            
        Raises:
            ModelLoadingError: If model or tokenizer loading fails
        """
        logger.info("Loading model from %s", self.config.model_name_or_path)
        
        try:
            tokenizer = self.tokenizer_loader.load(self.config)
            model = self._load_model()
            
            if self.config.use_lora:
                model = self.lora_configurator.prepare(model, self.config)
            
            return model, tokenizer
            
        except Exception as e:
            if not isinstance(e, (ModelLoadingError, TokenizerLoadingError)):
                raise ModelLoadingError(
                    f"Failed to load model from {self.config.model_name_or_path}"
                ) from e
            raise
    
    def _load_model(self) -> T:
        """Load the base model with the appropriate settings.
        
        Returns:
            Loaded model instance
            
        Raises:
            ModelLoadingError: If model loading fails
        """
        try:
            quantization_config = self.quant_config_factory.create(self.config)
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=self.config.trust_remote_code,
                use_auth_token=self.config.use_auth_token,
                torch_dtype=torch.bfloat16,
            )
            
            if self.config.gradient_checkpointing:
                model.gradient_checkpointing_enable()
                model.config.use_cache = False
            
            return model
            
        except Exception as e:
            raise ModelLoadingError(
                f"Failed to load model from {self.config.model_name_or_path}"
            ) from e

def create_model_loader(config: ModelConfig) -> ModelLoader[ModelType]:
    """Create a model loader with default dependencies.
    
    This factory function provides a clean interface for creating a ModelLoader
    with default configurations. It's the recommended way to create a ModelLoader
    instance for most use cases.
    
    Args:
        config: Model configuration containing all necessary parameters.
            Must include model_name_or_path and other required settings.
            
    Returns:
        ModelLoader: Configured ModelLoader instance ready to load models.
        
    Example:
        ```python
        config = ModelConfig(
            model_name_or_path="gpt2",
            use_lora=True,
            lora_rank=8
        )
        loader = create_model_loader(config)
        model, tokenizer = loader.load_model_and_tokenizer()
        ```
    """
    return ModelLoader[ModelType](config)
