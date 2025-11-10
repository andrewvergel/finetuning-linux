"""Tests for LoRA trainer."""
import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset

from finetuning_lora.config import ModelConfig, TrainingConfig, DataConfig
from finetuning_lora.training.trainer import LoRATrainer

@pytest.fixture
def sample_datasets():
    """Create sample datasets for testing."""
    train_data = {"input_ids": [[1, 2, 3], [4, 5, 6]], "attention_mask": [[1, 1, 1], [1, 1, 1]]}
    eval_data = {"input_ids": [[7, 8, 9]], "attention_mask": [[1, 1, 1]]}
    
    return {
        "train": Dataset.from_dict(train_data),
        "validation": Dataset.from_dict(eval_data)
    }

@pytest.fixture
def model_config():
    """Create a model configuration for testing."""
    return ModelConfig()

@pytest.fixture
def training_config(tmp_path):
    """Create a training configuration for testing with all required attributes."""
    config = TrainingConfig(output_dir=str(tmp_path / "test_output"))
    # Add missing attributes that trainer expects
    config.num_epochs = config.num_train_epochs
    config.per_device_batch_size = config.per_device_train_batch_size
    config.lr_scheduler = config.lr_scheduler_type
    config.save_strategy = "steps"
    config.eval_strategy = config.evaluation_strategy
    config.dataloader_num_workers = config.preprocessing_num_workers or 0
    config.remove_unused_columns = False
    config.report_to = ["tensorboard"]
    config.early_stopping = False
    return config

@pytest.fixture
def data_config():
    """Create a data configuration for testing."""
    return DataConfig()

@patch('finetuning_lora.training.trainer.setup_logging')
@patch('finetuning_lora.training.trainer.log_version_info')
@patch('finetuning_lora.training.trainer.set_seed')
def test_trainer_init(mock_set_seed, mock_log_version, mock_setup_logging, model_config, training_config, data_config):
    """Test trainer initialization."""
    # Initialize trainer
    trainer = LoRATrainer(model_config, training_config, data_config)
    
    # Verify initialization
    assert trainer.model_config == model_config
    assert trainer.training_config == training_config
    assert trainer.data_config == data_config
    assert trainer.model is None
    assert trainer.tokenizer is None
    assert trainer.trainer is None
    
    # Verify setup methods were called
    mock_setup_logging.assert_called_once()
    mock_log_version.assert_called_once()
    mock_set_seed.assert_called_once_with(training_config.seed)

@patch('finetuning_lora.training.trainer.setup_logging')
@patch('finetuning_lora.training.trainer.log_version_info')
@patch('finetuning_lora.training.trainer.set_seed')
@patch('finetuning_lora.models.builder.ModelBuilder')
@patch('finetuning_lora.training.trainer.Trainer')
@patch('finetuning_lora.training.trainer.TrainingArguments')
@patch('finetuning_lora.training.trainer.DataCollatorForLanguageModeling')
def test_train(
    mock_data_collator,
    mock_training_args,
    mock_trainer_class,
    mock_model_builder,
    mock_set_seed,
    mock_log_version,
    mock_setup_logging,
    model_config,
    training_config,
    data_config,
    sample_datasets
):
    """Test training process."""
    # Setup mocks
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_model_builder_instance = MagicMock()
    mock_model_builder_instance.load_model.return_value = (mock_model, mock_tokenizer)
    mock_model_builder.return_value = mock_model_builder_instance
    
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train.return_value = MagicMock(metrics={"train_loss": 0.5})
    mock_trainer_class.return_value = mock_trainer_instance
    
    mock_data_collator_instance = MagicMock()
    mock_data_collator.return_value = mock_data_collator_instance
    
    mock_training_args_instance = MagicMock()
    mock_training_args.return_value = mock_training_args_instance
    
    # Initialize trainer
    trainer = LoRATrainer(model_config, training_config, data_config)
    
    # Run training
    train_dataset, eval_dataset = sample_datasets["train"], sample_datasets["validation"]
    result_model, result_tokenizer = trainer.train(train_dataset, eval_dataset)
    
    # Verify TrainingArguments was created
    mock_training_args.assert_called_once()
    
    # Verify model builder was called
    mock_model_builder.assert_called_once_with(
        model_config=model_config,
        training_config=training_config
    )
    mock_model_builder_instance.load_model.assert_called_once()
    
    # Verify trainer was initialized with correct arguments
    mock_trainer_class.assert_called_once()
    call_args = mock_trainer_class.call_args
    assert call_args.kwargs["model"] == mock_model
    assert call_args.kwargs["tokenizer"] == mock_tokenizer
    assert call_args.kwargs["train_dataset"] == train_dataset
    assert call_args.kwargs["eval_dataset"] == eval_dataset
    
    # Verify training was called
    mock_trainer_instance.train.assert_called_once()
    
    # Verify save methods were called
    mock_trainer_instance.save_model.assert_called_once_with(training_config.output_dir)
    mock_tokenizer.save_pretrained.assert_called_once_with(training_config.output_dir)
    
    # Verify return values
    assert result_model == mock_model
    assert result_tokenizer == mock_tokenizer

@patch('finetuning_lora.training.trainer.setup_logging')
@patch('finetuning_lora.training.trainer.log_version_info')
@patch('finetuning_lora.training.trainer.set_seed')
def test_evaluate(mock_set_seed, mock_log_version, mock_setup_logging, model_config, training_config, data_config, sample_datasets):
    """Test evaluation."""
    # Setup
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.evaluate.return_value = {"eval_loss": 0.5, "eval_accuracy": 0.8}
    
    # Initialize trainer
    trainer = LoRATrainer(model_config, training_config, data_config)
    trainer.trainer = mock_trainer_instance
    
    # Run evaluation
    metrics = trainer.evaluate()
    
    # Verify evaluation was called
    assert "eval_loss" in metrics
    assert "eval_accuracy" in metrics
    assert metrics["eval_loss"] == 0.5
    mock_trainer_instance.evaluate.assert_called_once_with(eval_dataset=None)
    
    # Test with eval dataset
    eval_dataset = sample_datasets["validation"]
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    assert mock_trainer_instance.evaluate.call_count == 2
    mock_trainer_instance.evaluate.assert_called_with(eval_dataset=eval_dataset)

@patch('finetuning_lora.training.trainer.setup_logging')
@patch('finetuning_lora.training.trainer.log_version_info')
@patch('finetuning_lora.training.trainer.set_seed')
def test_save_model(mock_set_seed, mock_log_version, mock_setup_logging, model_config, training_config, data_config, tmp_path):
    """Test model saving."""
    # Initialize trainer
    trainer = LoRATrainer(model_config, training_config, data_config)
    
    # Create mock model and tokenizer
    trainer.model = MagicMock()
    trainer.tokenizer = MagicMock()
    
    # Test saving
    output_dir = tmp_path / "saved_model"
    trainer.save_model(str(output_dir))
    
    # Verify save methods were called
    trainer.model.save_pretrained.assert_called_once_with(str(output_dir))
    trainer.tokenizer.save_pretrained.assert_called_once_with(str(output_dir))
    
    # Verify directory was created
    assert output_dir.exists()

@patch('finetuning_lora.training.trainer.setup_logging')
@patch('finetuning_lora.training.trainer.log_version_info')
@patch('finetuning_lora.training.trainer.set_seed')
def test_save_model_peft(mock_set_seed, mock_log_version, mock_setup_logging, model_config, training_config, data_config, tmp_path):
    """Test model saving with PeftModel."""
    from peft import PeftModel
    
    # Initialize trainer
    trainer = LoRATrainer(model_config, training_config, data_config)
    
    # Create mock PeftModel
    mock_peft_model = MagicMock(spec=PeftModel)
    trainer.model = mock_peft_model
    trainer.tokenizer = MagicMock()
    
    # Test saving
    output_dir = tmp_path / "saved_peft_model"
    trainer.save_model(str(output_dir))
    
    # Verify save methods were called
    mock_peft_model.save_pretrained.assert_called_once_with(str(output_dir))
    trainer.tokenizer.save_pretrained.assert_called_once_with(str(output_dir))

@patch('finetuning_lora.training.trainer.setup_logging')
@patch('finetuning_lora.training.trainer.log_version_info')
@patch('finetuning_lora.training.trainer.set_seed')
def test_evaluate_without_trainer(mock_set_seed, mock_log_version, mock_setup_logging, model_config, training_config, data_config):
    """Test that evaluate raises error when trainer is not initialized."""
    # Initialize trainer without calling train()
    trainer = LoRATrainer(model_config, training_config, data_config)
    
    # Attempt to evaluate should raise ValueError
    with pytest.raises(ValueError, match="Trainer has not been initialized"):
        trainer.evaluate()

@patch('finetuning_lora.training.trainer.setup_logging')
@patch('finetuning_lora.training.trainer.log_version_info')
@patch('finetuning_lora.training.trainer.set_seed')
def test_save_model_without_model(mock_set_seed, mock_log_version, mock_setup_logging, model_config, training_config, data_config, tmp_path):
    """Test that save_model raises error when model is not initialized."""
    # Initialize trainer without model
    trainer = LoRATrainer(model_config, training_config, data_config)
    
    # Attempt to save should raise ValueError
    output_dir = tmp_path / "saved_model"
    with pytest.raises(ValueError, match="Model and tokenizer must be initialized"):
        trainer.save_model(str(output_dir))
