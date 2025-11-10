"""Tests for data processor."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from finetuning_lora.config.data import DataConfig
from finetuning_lora.data.processor import DataProcessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return [
        {"input": "Hello", "output": "Hi there!"},
        {"input": "How are you?", "output": "I'm doing well, thank you!"}
    ]

@pytest.fixture
def temp_jsonl(sample_data):
    """Create a temporary JSONL file for testing."""
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
        f.flush()
        yield f.name
        Path(f.name).unlink()

def test_data_processor_init(temp_jsonl):
    """Test DataProcessor initialization."""
    config = DataConfig(train_path=temp_jsonl)
    processor = DataProcessor(config)
    assert processor.config == config

def test_validate_example(temp_jsonl):
    """Test example validation."""
    config = DataConfig(train_path=temp_jsonl)
    processor = DataProcessor(config)
    
    valid_example = {"input": "test", "output": "test"}
    invalid_example = {"input": "", "output": "test"}
    
    assert processor._validate_example(valid_example) is True
    assert processor._validate_example(invalid_example) is False

def test_load_datasets(temp_jsonl):
    """Test dataset loading and splitting."""
    config = DataConfig(train_path=temp_jsonl)
    processor = DataProcessor(config)
    
    # Test with validation split
    datasets = processor.load_datasets()
    assert "train" in datasets
    assert "validation" in datasets
    assert len(datasets["train"]) > 0
    assert len(datasets["validation"]) > 0

@patch('transformers.AutoTokenizer.from_pretrained')
def test_format_example(mock_tokenizer, temp_jsonl):
    """Test example formatting with tokenizer."""
    # Mock tokenizer
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.apply_chat_template.return_value = "<|system|>\nHello<|endoftext|>"
    mock_tokenizer.return_value = mock_tokenizer_instance
    
    config = DataConfig(train_path=temp_jsonl)
    processor = DataProcessor(config, tokenizer=mock_tokenizer_instance)
    
    example = {"input": "Hello", "output": "Hi there!"}
    formatted = processor.format_example(example)
    assert "text" in formatted
    assert isinstance(formatted["text"], str)
    mock_tokenizer_instance.apply_chat_template.assert_called_once()

def test_tokenize_function(temp_jsonl):
    """Test tokenization function."""
    # Create a mock tokenizer that returns tokenized output
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.pad_token_id = 0
    mock_tokenizer_instance.eos_token_id = 1
    
    # Configure the tokenizer call to return the expected format
    # The tokenizer is called with text and kwargs, and should return a dict
    mock_tokenizer_instance.return_value = {
        "input_ids": [[1, 2, 3, 4]],
        "attention_mask": [[1, 1, 1, 1]]
    }
    
    config = DataConfig(train_path=temp_jsonl, max_length=512, padding_side="right")
    processor = DataProcessor(config, tokenizer=mock_tokenizer_instance)
    
    examples = {"text": ["Test example"]}
    tokenized = processor.tokenize_function(examples)
    
    assert "input_ids" in tokenized
    assert "attention_mask" in tokenized
    # Verify tokenizer was called with the text
    mock_tokenizer_instance.assert_called_once()
    call_args = mock_tokenizer_instance.call_args
    assert call_args[0][0] == examples["text"]
