"""Tests for model configuration."""
import pytest
from finetuning_lora.config.model import ModelConfig, LoraBiasType, QuantizationType, ComputeDtype

def test_model_config_defaults():
    """Test model configuration default values."""
    config = ModelConfig()
    assert config.model_name_or_path == "Qwen/Qwen2.5-7B-Instruct"
    assert config.trust_remote_code is True
    assert config.use_lora is True
    assert config.lora_rank == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.05
    assert config.lora_bias == LoraBiasType.NONE

def test_quantization_config():
    """Test quantization configuration."""
    config = ModelConfig(
        load_in_4bit=True,
        load_in_8bit=False,  # Explicitly set to False to avoid conflicts
        bnb_4bit_quant_type=QuantizationType.NF4,
        bnb_4bit_compute_dtype=ComputeDtype.BFLOAT16,
        use_nested_quant=True
    )
    quant_config = config.to_quantization_config()
    assert quant_config["load_in_4bit"] is True
    assert quant_config["bnb_4bit_quant_type"] == "nf4"
    assert quant_config["bnb_4bit_compute_dtype"] == "bfloat16"
    assert quant_config["bnb_4bit_use_double_quant"] is True

@pytest.mark.parametrize("bias_type", ["none", "all", "lora_only"])
def test_lora_bias_types(bias_type):
    """Test different LoRA bias types."""
    config = ModelConfig(lora_bias=LoraBiasType(bias_type))
    peft_config = config.to_peft_config()
    assert peft_config["bias"] == bias_type
