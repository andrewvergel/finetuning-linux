"""Tests for model builder."""
import pytest
from unittest.mock import patch, MagicMock
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

from finetuning_lora.config import ModelConfig, TrainingConfig
from finetuning_lora.models.builder import ModelBuilder

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock(spec=AutoModelForCausalLM)
    model.config = {}
    return model

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock(spec=AutoTokenizer)
    tokenizer.pad_token = None
    tokenizer.eos_token = "<|endoftext|>"
    return tokenizer

def test_model_builder_init():
    """Test ModelBuilder initialization."""
    # Explicitly disable quantization to avoid validation issues
    model_config = ModelConfig(load_in_4bit=False, load_in_8bit=False)
    training_config = TrainingConfig()
    
    builder = ModelBuilder(model_config, training_config)
    
    assert builder.model_config == model_config
    assert builder.training_config == training_config
    assert builder.device.type in ("cuda", "cpu")

@patch('finetuning_lora.models.builder.log_version_info')
@patch('finetuning_lora.models.builder.AutoModelForCausalLM.from_pretrained')
@patch('finetuning_lora.models.builder.AutoTokenizer.from_pretrained')
def test_load_model(mock_tokenizer_from_pretrained, mock_model_from_pretrained, mock_log_version_info):
    """Test model loading."""
    # Setup mocks - create a mock tokenizer to return
    mock_tokenizer_obj = MagicMock(spec=AutoTokenizer)
    mock_tokenizer_obj.pad_token = None
    mock_tokenizer_obj.eos_token = "<|endoftext|>"
    
    mock_model_instance = MagicMock()
    mock_model_instance.config = MagicMock()
    mock_model_instance.config.use_cache = True
    mock_model_instance.gradient_checkpointing_enable = MagicMock()
    mock_model_instance.eval = MagicMock(return_value=mock_model_instance)
    mock_model_from_pretrained.return_value = mock_model_instance
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer_obj
    
    # Test with config that disables quantization and LoRA for basic load test
    model_config = ModelConfig(
        load_in_4bit=False, 
        load_in_8bit=False,
        use_lora=False
    )
    builder = ModelBuilder(model_config)
    model, tokenizer = builder.load_model()
    
    assert model is not None
    assert tokenizer is not None
    mock_model_from_pretrained.assert_called_once()
    mock_tokenizer_from_pretrained.assert_called_once()

@patch('finetuning_lora.models.builder.log_version_info')
@patch('finetuning_lora.models.builder.get_peft_model')
@patch('finetuning_lora.models.builder.prepare_model_for_kbit_training')
@patch('finetuning_lora.models.builder.AutoModelForCausalLM.from_pretrained')
@patch('finetuning_lora.models.builder.AutoTokenizer.from_pretrained')
def test_apply_lora(
    mock_tokenizer_from_pretrained,
    mock_model_from_pretrained,
    mock_prepare_model,
    mock_get_peft_model,
    mock_log_version_info
):
    """Test LoRA application in load_model."""
    # Setup mocks
    mock_tokenizer_obj = MagicMock(spec=AutoTokenizer)
    mock_tokenizer_obj.pad_token = None
    mock_tokenizer_obj.eos_token = "<|endoftext|>"
    
    mock_model_instance = MagicMock()
    mock_model_instance.config = {}
    mock_model_instance.gradient_checkpointing_enable = MagicMock()
    mock_model_instance.eval = MagicMock(return_value=mock_model_instance)
    
    mock_peft_model = MagicMock()
    mock_peft_model.print_trainable_parameters = MagicMock()  # Mock this method to avoid hanging
    mock_peft_model.config = MagicMock()
    mock_peft_model.config.use_cache = True
    mock_peft_model.gradient_checkpointing_enable = MagicMock()
    mock_peft_model.eval = MagicMock(return_value=mock_peft_model)
    
    # Configure the mocks
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer_obj
    mock_model_from_pretrained.return_value = mock_model_instance
    mock_prepare_model.return_value = mock_model_instance
    mock_get_peft_model.return_value = mock_peft_model
    
    # Test with LoRA enabled, explicitly disable quantization
    model_config = ModelConfig(
        use_lora=True, 
        lora_rank=8, 
        lora_alpha=16,
        load_in_4bit=False,
        load_in_8bit=False
    )
    builder = ModelBuilder(model_config)
    
    # Call load_model which should apply LoRA
    model, tokenizer = builder.load_model()
    
    # Verify LoRA was applied
    assert model == mock_peft_model
    mock_get_peft_model.assert_called_once()
    # Verify print_trainable_parameters was called
    mock_peft_model.print_trainable_parameters.assert_called_once()
    # Verify the model was not prepared for k-bit training (quantization disabled)
    mock_prepare_model.assert_not_called()

@patch('finetuning_lora.models.builder.BitsAndBytesConfig')
def test_quantization_config(mock_bnb_config):
    """Test quantization configuration."""
    # Setup
    import torch
    # Ensure torch has the bfloat16 attribute for the test
    if not hasattr(torch, 'bfloat16'):
        torch.bfloat16 = MagicMock()
    
    mock_config_instance = MagicMock()
    mock_bnb_config.return_value = mock_config_instance
    
    # Test with 4-bit quantization, explicitly disable 8-bit
    model_config = ModelConfig(load_in_4bit=True, load_in_8bit=False)
    builder = ModelBuilder(model_config)
    
    config = builder._get_quantization_config()
    
    assert config is not None
    assert config == mock_config_instance
    mock_bnb_config.assert_called_once()
    # Verify BitsAndBytesConfig was called with correct parameters
    call_kwargs = mock_bnb_config.call_args.kwargs
    assert call_kwargs['load_in_4bit'] is True
    assert call_kwargs['load_in_8bit'] is False
    # Verify the quant_type and compute_dtype are strings (enum values)
    assert isinstance(call_kwargs['bnb_4bit_quant_type'], str)
    assert call_kwargs['bnb_4bit_compute_dtype'] is not None

def test_save_model(tmp_path, mock_model, mock_tokenizer):
    """Test model saving."""
    # Explicitly disable quantization to avoid validation issues
    model_config = ModelConfig(load_in_4bit=False, load_in_8bit=False)
    builder = ModelBuilder(model_config)
    output_dir = tmp_path / "test_model"
    
    # Ensure the mock model has save_pretrained method
    mock_model.save_pretrained = MagicMock()
    mock_tokenizer.save_pretrained = MagicMock()
    
    # Test saving
    builder.save_model(mock_model, mock_tokenizer, str(output_dir))
    
    # Verify the model's save_pretrained was called
    mock_model.save_pretrained.assert_called_once_with(str(output_dir))
    mock_tokenizer.save_pretrained.assert_called_once_with(str(output_dir))
