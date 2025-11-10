#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for RTX 4060 Ti
Version: 2.0.0
Author: Refactored to use structured configuration classes
Optimized for: RTX 4060 Ti (16GB VRAM)

Changelog:
Version: 2.0.0
- Refactored to use structured configuration classes (ModelConfig, TrainingConfig, DataConfig)
- Uses ModelBuilder for model loading
- Uses DataProcessor for data processing
- Maintains all previous functionality while improving code organization
- Better separation of concerns and testability

Version: 1.2.0
- bf16 por defecto (más estable en Ada/Lovelace)
- Packing forzado (opcional por .env)
- Gradient Checkpointing + use_cache=False
- EarlyStopping + eval/save por pasos
- Opción QLoRA 4-bit activable por .env sin tocar el código
"""

import os
import sys
import math
import json
import logging
from typing import List, Optional
from datetime import datetime
from pathlib import Path

# Add src directory to Python path to enable imports
# This allows the script to be run from any directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if not SRC_DIR.exists():
    raise RuntimeError(
        f"Error: Could not find 'src' directory at {SRC_DIR}. "
        f"Please ensure you're running this script from the project root directory."
    )
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from datasets import Dataset, concatenate_datasets
from packaging import version
from transformers import TrainingArguments, EarlyStoppingCallback, DataCollatorForLanguageModeling
from trl import SFTTrainer

# Import structured components
from finetuning_lora.config import (
    load_env_file,
    load_model_config,
    load_training_config,
    load_data_config,
)
from finetuning_lora.models.builder import ModelBuilder
from finetuning_lora.data.processor import DataProcessor
from finetuning_lora.utils.logging import setup_logging, log_version_info

SCRIPT_VERSION = "2.0.0"
SCRIPT_NAME = "finetune_lora.py"

# Check required versions
import transformers
import peft

REQUIRED_TRANSFORMERS = "4.40.0"
REQUIRED_PEFT = "0.10.0"

if version.parse(transformers.__version__) < version.parse(REQUIRED_TRANSFORMERS):
    raise RuntimeError(
        f"transformers>={REQUIRED_TRANSFORMERS} is required. "
        f"Found {transformers.__version__}. Please upgrade: pip install --upgrade transformers"
    )

if version.parse(peft.__version__) < version.parse(REQUIRED_PEFT):
    raise RuntimeError(
        f"peft>={REQUIRED_PEFT} is required. Found {peft.__version__}. "
        f"Please upgrade: pip install --upgrade peft"
    )

# Load environment variables
load_env_file()

# Enable memory optimizations
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_DISABLE_MULTIPROCESSING", "1")

# Evaluation fallback prompts
EVAL_FALLBACK_PROMPTS = [
    {
        "system": "Eres un asistente experto en procesos internos.",
        "user": "Dame los pasos para conciliar pagos de los lunes.",
        "expected": "1) Exporta el CSV del banco...",
    }
]


def run_eval(model, tokenizer, device, eval_prompts):
    """Run evaluation on sample prompts.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        device: Device to run inference on
        eval_prompts: List of evaluation prompts
    """
    if not eval_prompts:
        logging.warning(">> No evaluation prompts available. Skipping quick evaluation.")
        return

    logging.info(">> Running quick evaluation on %d sample prompts...", len(eval_prompts))
    model.eval()

    for sample in eval_prompts:
        # Build messages for chat template
        messages = [
            {
                "role": "system",
                "content": sample.get("system", "Eres un asistente útil y conciso."),
            },
            {"role": "user", "content": sample.get("user", "")},
        ]

        # Generate prompt with chat template
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize and generate
        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,  # Default eval max new tokens
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Decode and extract assistant response
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        user_content = messages[-1]["content"]
        assistant_response = decoded.split(user_content)[-1].strip()

        logging.info("[Eval] User: %s", sample.get("user", ""))
        logging.info("[Eval] Expected: %s", sample.get("expected", "No expected output"))
        logging.info("[Eval] Model: %s", assistant_response)
        logging.info("-" * 80)


def expand_dataset_if_needed(dataset: Dataset, min_examples: int, seed: int = 42) -> Dataset:
    """Expand dataset if it's too small by repeating examples.
    
    Args:
        dataset: Dataset to expand
        min_examples: Minimum number of examples required
        seed: Random seed for shuffling
        
    Returns:
        Expanded dataset
    """
    if len(dataset) >= min_examples:
        return dataset

    repeat_factor = max(1, math.ceil(min_examples / len(dataset)))
    expanded = concatenate_datasets([dataset] * repeat_factor).shuffle(seed=seed)
    logging.info(
        ">> Training dataset expanded: %dx -> %d examples", repeat_factor, len(expanded)
    )
    return expanded


def calculate_dataset_stats(dataset: Dataset, tokenizer, max_seq_len: int, sample_size: int = 64):
    """Calculate statistics about the dataset.
    
    Args:
        dataset: Dataset to analyze
        tokenizer: Tokenizer for token counting
        max_seq_len: Maximum sequence length
        sample_size: Number of samples to analyze
        
    Returns:
        Dictionary with dataset statistics
    """
    sample_size = min(sample_size, len(dataset))
    token_lengths = []
    total_chars = 0

    for i in range(sample_size):
        text = dataset[i]["text"]
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        token_lengths.append(len(tokens["input_ids"][0]))
        total_chars += len(text)

    stats = {
        "avg_tokens": sum(token_lengths) / len(token_lengths) if token_lengths else 0,
        "max_tokens": max(token_lengths) if token_lengths else 0,
        "min_tokens": min(token_lengths) if token_lengths else 0,
        "avg_chars": total_chars / sample_size if sample_size > 0 else 0,
        "total_examples": len(dataset),
    }

    # Calculate approximate total tokens
    stats["approx_total_tokens"] = stats["avg_tokens"] * len(dataset)

    return stats


def should_use_packing(
    dataset: Dataset, stats: dict, max_seq_len: int, force_packing: bool, min_examples: int = 64
) -> bool:
    """Determine if packing should be used.
    
    Args:
        dataset: Training dataset
        stats: Dataset statistics
        max_seq_len: Maximum sequence length
        force_packing: Whether packing is forced via config
        min_examples: Minimum examples required for packing
        
    Returns:
        Whether to use packing
    """
    if not force_packing:
        return False

    min_tokens_for_packing = max_seq_len * 4

    if len(dataset) < min_examples:
        logging.warning(
            "Forced packing disabled: dataset too small (%d < %d examples)",
            len(dataset),
            min_examples,
        )
        return False

    if stats["approx_total_tokens"] < min_tokens_for_packing:
        logging.warning(
            "Forced packing disabled: insufficient estimated tokens (%.0f < %d)",
            stats["approx_total_tokens"],
            min_tokens_for_packing,
        )
        return False

    return True


def main():
    """Main training function."""
    # Setup logging
    log_path = setup_logging()
    log_version_info(SCRIPT_NAME, SCRIPT_VERSION)

    # Enable SDP kernels if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    # Load configurations from environment
    model_config = load_model_config()
    training_config = load_training_config()
    data_config = load_data_config()

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(">> Device detected: %s", device)

    # Validate dataset exists
    if not Path(data_config.train_path).exists():
        logging.error("❌ Error: dataset %s not found.", data_config.train_path)
        return

    logging.info(">> Loading dataset from: %s", data_config.train_path)

    # Load model and tokenizer using ModelBuilder
    logging.info(">> Loading model: %s", model_config.model_name_or_path)
    model_builder = ModelBuilder(model_config=model_config, training_config=training_config)
    model, tokenizer = model_builder.load_model()

    # Ensure tokenizer has pad_token set (required for SFTTrainer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info(">> Set tokenizer.pad_token to eos_token")
    
    # Ensure tokenizer has proper configuration for padding
    # This is critical for SFTTrainer to work correctly
    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logging.info(">> Set tokenizer.pad_token_id to eos_token_id")
    
    # Verify tokenizer configuration
    logging.info(f">> Tokenizer pad_token: {tokenizer.pad_token}")
    logging.info(f">> Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    logging.info(f">> Tokenizer eos_token: {tokenizer.eos_token}")
    logging.info(f">> Tokenizer eos_token_id: {tokenizer.eos_token_id}")

    logging.info(">> Model loaded on %s", device)

    # Determine max sequence length
    max_seq_len = training_config.max_seq_length
    # Check model and tokenizer limits
    model_max_len = getattr(model.config, "max_position_embeddings", None)
    tokenizer_max_len = getattr(tokenizer, "model_max_length", None)
    
    # Use the minimum of all available limits
    candidates = [max_seq_len]
    if model_max_len:
        candidates.append(model_max_len)
    if tokenizer_max_len and tokenizer_max_len != float('inf'):
        candidates.append(tokenizer_max_len)
    
    max_seq_len = min(candidates) if candidates else 512
    max_seq_len = min(max_seq_len, 1024)  # Cap at 1024 for memory efficiency
    training_config.max_seq_length = max_seq_len
    logging.info(">> Using max sequence length: %d tokens", max_seq_len)

    # Set model to training mode
    model.train()

    # Enable memory optimizations
    model.config.use_cache = False
    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logging.info(">> Gradient checkpointing enabled with reentrant=False")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(
        f">> Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # Process data using DataProcessor
    logging.info(">> Processing dataset...")
    data_processor = DataProcessor(config=data_config, tokenizer=tokenizer)
    train_dataset, eval_dataset = data_processor.prepare_datasets_for_sft(
        tokenizer=tokenizer, shuffle=True, seed=data_config.shuffle_seed
    )

    # Expand dataset if needed
    # For very small datasets, expand more aggressively to ensure sufficient training data
    original_dataset_size = len(train_dataset)
    dataset_min_examples = int(os.getenv("FT_DATASET_MIN_EXAMPLES", "100"))
    
    # For very small datasets (< 30 original examples), expand to at least 200 examples
    # More repetitions = more opportunities to memorize patterns
    if original_dataset_size < 30:
        dataset_min_examples = max(dataset_min_examples, 200)
        logging.info(">> Very small original dataset (%d examples) - expanding to at least %d examples for better memorization", 
                    original_dataset_size, dataset_min_examples)
    
    train_dataset = expand_dataset_if_needed(
        train_dataset, dataset_min_examples, seed=data_config.shuffle_seed
    )

    logging.info(
        ">> Train samples: %d | Eval samples: %d",
        len(train_dataset),
        len(eval_dataset) if eval_dataset else 0,
    )

    # Calculate dataset statistics
    stats = calculate_dataset_stats(train_dataset, tokenizer, max_seq_len)

    # Log dataset statistics
    logging.info("📊 Dataset statistics:")
    logging.info(f"  - Examples: {stats['total_examples']}")
    logging.info(
        f"  - Tokens/example: {stats['avg_tokens']:.1f} "
        f"(min: {stats['min_tokens']}, max: {stats['max_tokens']})"
    )
    logging.info(f"  - Characters/example: {stats['avg_chars']:.1f}")
    logging.info(f"  - Estimated total tokens: {stats['approx_total_tokens']:,.0f}")

    # Adjust hyperparameters for small datasets to improve deep learning
    # Small datasets (< 200 examples) need more aggressive learning to memorize patterns
    dataset_size = len(train_dataset)
    is_small_dataset = dataset_size < 200
    is_very_small_dataset = dataset_size < 150  # Very small datasets need even more aggressive settings
    
    if is_very_small_dataset:
        logging.info("🔧 Very small dataset detected (%d examples) - applying ULTRA-AGGRESSIVE hyperparameters for deep memorization", dataset_size)
        
        # ULTRA-AGGRESSIVE learning rate for memorization (5e-5 to 6e-5)
        # Higher LR is critical for memorizing specific patterns
        original_lr = training_config.learning_rate
        if original_lr < 5e-5:
            training_config.learning_rate = 5e-5
            logging.info(f"  - Learning rate increased ULTRA-AGGRESSIVELY: {original_lr:.2e} -> {training_config.learning_rate:.2e}")
        
        # More epochs for very small datasets (12-15 for maximum memorization)
        if training_config.num_train_epochs < 12:
            original_epochs = training_config.num_train_epochs
            training_config.num_train_epochs = 12
            logging.info(f"  - Epochs increased significantly: {original_epochs} -> {training_config.num_train_epochs}")
        
        # Minimal warmup for very small datasets (2% to start learning immediately)
        if training_config.warmup_ratio > 0.02:
            original_warmup = training_config.warmup_ratio
            training_config.warmup_ratio = 0.02
            logging.info(f"  - Warmup ratio reduced to absolute minimum: {original_warmup} -> {training_config.warmup_ratio} (immediate learning)")
        
        # Disable NEFTune completely for very small datasets (it interferes with memorization)
        if training_config.neftune_noise_alpha is not None and training_config.neftune_noise_alpha > 0:
            original_neftune = training_config.neftune_noise_alpha
            training_config.neftune_noise_alpha = None
            logging.info(f"  - NEFTune DISABLED: {original_neftune} -> None (maximize memorization)")
        
        # Minimize weight decay for maximum memorization (almost no regularization)
        if training_config.weight_decay > 0.001:
            original_wd = training_config.weight_decay
            training_config.weight_decay = 0.001
            logging.info(f"  - Weight decay minimized: {original_wd} -> {training_config.weight_decay} (maximum memorization)")
        
        # Reduce LoRA dropout for better memorization (less regularization in LoRA layers)
        if hasattr(model_config, 'lora_dropout') and model_config.lora_dropout > 0.01:
            original_lora_dropout = model_config.lora_dropout
            model_config.lora_dropout = 0.01
            logging.info(f"  - LoRA dropout reduced: {original_lora_dropout} -> {model_config.lora_dropout} (better memorization)")
        
        # Increase early stopping patience significantly (allow more training)
        if training_config.early_stopping_patience < 12:
            original_patience = training_config.early_stopping_patience
            training_config.early_stopping_patience = 12
            logging.info(f"  - Early stopping patience increased significantly: {original_patience} -> {training_config.early_stopping_patience}")
        
        logging.info("✅ ULTRA-AGGRESSIVE hyperparameters applied for very small dataset deep memorization")
        
    elif is_small_dataset:
        logging.info("🔧 Small dataset detected (%d examples) - adjusting hyperparameters for deeper learning", dataset_size)
        
        # Increase learning rate for better pattern learning (3.5e-5 to 4e-5 for small datasets)
        original_lr = training_config.learning_rate
        if original_lr < 3.5e-5:
            training_config.learning_rate = 4e-5
            logging.info(f"  - Learning rate increased: {original_lr:.2e} -> {training_config.learning_rate:.2e}")
        
        # Increase epochs for better convergence (8-10 for small datasets)
        if training_config.num_train_epochs < 8:
            original_epochs = training_config.num_train_epochs
            training_config.num_train_epochs = 8
            logging.info(f"  - Epochs increased: {original_epochs} -> {training_config.num_train_epochs}")
        
        # Reduce warmup ratio to reach effective learning rate faster (5% instead of 10%)
        if training_config.warmup_ratio > 0.05:
            original_warmup = training_config.warmup_ratio
            training_config.warmup_ratio = 0.05
            logging.info(f"  - Warmup ratio reduced: {original_warmup} -> {training_config.warmup_ratio} (faster learning)")
        
        # Reduce NEFTune for small datasets (it can interfere with deep memorization)
        if training_config.neftune_noise_alpha is not None and training_config.neftune_noise_alpha > 0:
            original_neftune = training_config.neftune_noise_alpha
            # Reduce NEFTune noise for small datasets (0.05 instead of 0.1)
            training_config.neftune_noise_alpha = 0.05
            logging.info(f"  - NEFTune noise reduced: {original_neftune} -> {training_config.neftune_noise_alpha} (better memorization)")
        
        # Increase early stopping patience for small datasets
        if training_config.early_stopping_patience < 7:
            original_patience = training_config.early_stopping_patience
            training_config.early_stopping_patience = 7
            logging.info(f"  - Early stopping patience increased: {original_patience} -> {training_config.early_stopping_patience}")
        
        logging.info("✅ Hyperparameters adjusted for small dataset deep learning")
    else:
        logging.info("📊 Standard dataset size - using default hyperparameters optimized for generalization")

    # Determine if packing should be used
    force_packing = bool(os.getenv("FT_FORCE_PACKING", "false").lower() in ("true", "1", "yes"))
    use_packing = should_use_packing(train_dataset, stats, max_seq_len, force_packing)
    logging.info(f"  - Packing: {'ENABLED' if use_packing else 'DISABLED'}")

    if stats["max_tokens"] > max_seq_len * 0.9:
        logging.warning(
            "⚠️  Warning! Some examples are close to the context limit (%d tokens)",
            max_seq_len,
        )

    # Prepare evaluation prompts
    eval_prompts = []
    if eval_dataset:
        eval_sample_size = int(os.getenv("FT_EVAL_SAMPLE_SIZE", "3"))
        sample_n = min(eval_sample_size, len(eval_dataset))
        for i in range(sample_n):
            row = eval_dataset[i]
            # Extract system, input, output from formatted text if needed
            # For now, use fallback prompts
            eval_prompts = EVAL_FALLBACK_PROMPTS
            break
    if not eval_prompts:
        eval_prompts = EVAL_FALLBACK_PROMPTS

    # Prepare training arguments
    training_args_dict = training_config.to_training_args()
    
    # Disable bf16 when using QLoRA (QLoRA handles its own quantization)
    if model_config.load_in_4bit:
        training_args_dict["bf16"] = False
        training_args_dict["fp16"] = False
        logging.info(">> QLoRA enabled: bf16/fp16 disabled (using 4-bit quantization)")
    
    # CRITICAL FIX: For SFTTrainer with non-packed datasets, we MUST set remove_unused_columns=True
    # When remove_unused_columns=False, the 'text' field remains in the dataset after tokenization,
    # causing the data collator to try to pad both tokenized fields (input_ids) and the text field (string),
    # resulting in "excessive nesting" error. SFTTrainer will automatically remove the 'text' field
    # after tokenization when remove_unused_columns=True.
    if not use_packing:
        training_args_dict["remove_unused_columns"] = True
        logging.info(">> Set remove_unused_columns=True for SFTTrainer (required for non-packed datasets)")
    else:
        # For packed datasets, we can keep remove_unused_columns=False if needed
        logging.info(">> Using remove_unused_columns=%s (packing enabled)", training_config.remove_unused_columns)
    
    # Update with eval dataset configuration
    if eval_dataset and len(eval_dataset) > 0:
        training_args_dict["evaluation_strategy"] = "steps"
        training_args_dict["eval_steps"] = training_config.eval_steps
        training_args_dict["load_best_model_at_end"] = True
        training_args_dict["metric_for_best_model"] = "eval_loss"
    else:
        training_args_dict["evaluation_strategy"] = "no"
        training_args_dict["load_best_model_at_end"] = False

    training_args = TrainingArguments(**training_args_dict)
    logging.info(">> Training configuration set")

    # Prepare callbacks
    callbacks = []
    if training_config.early_stopping and eval_dataset and len(eval_dataset) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_config.early_stopping_patience,
                early_stopping_threshold=training_config.early_stopping_threshold,
            )
        )

    # Verify dataset structure before training
    logging.info(">> Verifying dataset structure...")
    sample_example = train_dataset[0]
    if "text" not in sample_example:
        raise ValueError(
            f"Dataset missing 'text' field. Found keys: {list(sample_example.keys())}"
        )
    
    # Ensure text is a plain string (not nested)
    text_value = sample_example["text"]
    if isinstance(text_value, list):
        raise ValueError(
            f"Dataset 'text' field contains a list (nested structure). "
            f"This will cause tokenization errors. First item type: {type(text_value[0]) if text_value else 'empty list'}"
        )
    if not isinstance(text_value, str):
        raise ValueError(
            f"Dataset 'text' field must be a string, got {type(text_value)}. Value: {text_value}"
        )
    if not text_value.strip():
        raise ValueError("Dataset 'text' field contains empty string")
    
    logging.info(f">> Dataset structure verified: text field length = {len(text_value)} chars")
    logging.info(f">> Sample text preview: {text_value[:100]}...")
    
    # Additional validation: Check multiple examples to ensure consistency
    logging.info(">> Validating multiple dataset examples...")
    for i in range(min(5, len(train_dataset))):
        example = train_dataset[i]
        if "text" not in example:
            raise ValueError(f"Example {i} missing 'text' field")
        text = example["text"]
        if not isinstance(text, str):
            raise ValueError(f"Example {i} has non-string text field: {type(text)}")
        if isinstance(text, list):
            raise ValueError(f"Example {i} has list in text field (nested structure)")
        if not text.strip():
            raise ValueError(f"Example {i} has empty text field")
    logging.info(f">> Validated {min(5, len(train_dataset))} examples - all have proper string text fields")
    
    # Test tokenization on a sample to ensure it works correctly
    logging.info(">> Testing tokenization on sample text...")
    try:
        test_tokenized = tokenizer(
            text_value,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt"
        )
        logging.info(f">> Tokenization test successful: input_ids shape = {test_tokenized['input_ids'].shape}")
        logging.info(f">> Tokenization test: attention_mask shape = {test_tokenized['attention_mask'].shape}")
    except Exception as e:
        logging.error(f">> Tokenization test failed: {e}")
        raise ValueError(f"Tokenizer test failed - this will cause training to fail: {e}") from e

    # Initialize SFTTrainer
    logging.info("🚀 Initializing SFTTrainer...")
    logging.info(f">> Packing: {use_packing}")
    logging.info(f">> Dataset text field: {training_config.dataset_text_field}")
    logging.info(f">> Max sequence length: {max_seq_len}")
    logging.info(f">> Tokenizer pad_token: {tokenizer.pad_token}")
    logging.info(f">> Tokenizer pad_token_id: {tokenizer.pad_token_id}")

    sft_trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset if eval_dataset and len(eval_dataset) > 0 else None,
        "args": training_args,
        "dataset_text_field": training_config.dataset_text_field,
        "max_seq_length": max_seq_len,
        "packing": use_packing,
        "callbacks": callbacks,
    }

    # Add optional SFTTrainer parameters
    if training_config.neftune_noise_alpha is not None:
        sft_trainer_kwargs["neftune_noise_alpha"] = training_config.neftune_noise_alpha

    if training_config.dataset_num_proc is not None:
        sft_trainer_kwargs["dataset_num_proc"] = training_config.dataset_num_proc

    # Ensure tokenizer padding is configured correctly
    # SFTTrainer needs the tokenizer to handle padding properly
    tokenizer.padding_side = "right"  # Right padding is standard for causal LM
    tokenizer.truncation_side = "right"  # Truncate from the right
    
    logging.info(f">> Tokenizer padding_side: {tokenizer.padding_side}")
    logging.info(f">> Tokenizer truncation_side: {tokenizer.truncation_side}")
    
    # NOTE: SFTTrainer handles tokenization internally
    # We should NOT provide a data_collator when using SFTTrainer with text fields
    # SFTTrainer will create its own data collator that handles tokenization
    # Providing a custom data collator can interfere with SFTTrainer's tokenization
    # Only use a custom data collator if you're pre-tokenizing the data yourself

    trainer = SFTTrainer(**sft_trainer_kwargs)

    # Verify model is ready for training
    logging.info(f">> Model device: {next(trainer.model.parameters()).device}")
    logging.info(f">> Model training mode: {trainer.model.training}")

    # Create output directory
    os.makedirs(training_config.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Log configuration summary
    logging.info("\n" + "=" * 80)
    logging.info("🚀 STARTING TRAINING")
    logging.info("=" * 80)
    logging.info("📋 Configuration:")
    logging.info(f"   - Model: {model_config.model_name_or_path}")
    logging.info(
        f"   - Batch size: {training_args.per_device_train_batch_size} "
        f"(x{training_args.gradient_accumulation_steps})"
    )
    logging.info(f"   - Max length: {max_seq_len} tokens")
    logging.info(f"   - Epochs: {training_args.num_train_epochs}")
    logging.info(
        f"   - Dataset size: {len(train_dataset)} training, "
        f"{len(eval_dataset) if eval_dataset else 0} validation"
    )
    logging.info(f"   - Learning rate: {training_args.learning_rate}")
    logging.info(f"   - Weight decay: {training_args.weight_decay}")
    logging.info(
        f"   - LoRA r={model_config.lora_rank}, alpha={model_config.lora_alpha}, "
        f"dropout={model_config.lora_dropout}"
    )
    logging.info(f"   - QLoRA: {'Yes' if model_config.load_in_4bit else 'No'}")
    logging.info(f"   - Packing: {'Yes' if use_packing else 'No'}")
    logging.info("=" * 80 + "\n")

    # Training with error handling
    try:
        logging.info("🏋️ Starting training...")
        train_result = trainer.train()

        # Save training metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        # Final evaluation
        eval_metrics = {}
        if eval_dataset and len(eval_dataset) > 0:
            logging.info("📊 Evaluating final model...")
            eval_metrics = trainer.evaluate()
            metrics.update({"eval_" + k: v for k, v in eval_metrics.items()})

        # Save metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        logging.info("✅ Training completed successfully!")
        logging.info("📊 Evaluation metrics:")
        for k, v in eval_metrics.items():
            logging.info(f"   - {k}: {v:.4f}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logging.error(
                "❌ Error: Insufficient memory. Try reducing batch size or sequence length."
            )
        raise
    except Exception as e:
        logging.error(f"❌ Error during training: {str(e)}")
        # Save model despite error if possible
        try:
            trainer.save_model(training_config.output_dir + "_crashed")
            logging.info(f"Model saved to {training_config.output_dir}_crashed")
        except Exception:
            pass
        raise

    # Save model
    logging.info(">> Saving model (adapters + tokenizer)...")
    trainer.model.save_pretrained(training_config.output_dir)
    tokenizer.save_pretrained(training_config.output_dir)

    # Save training info
    training_info = {
        "script_version": SCRIPT_VERSION,
        "model_id": model_config.model_name_or_path,
        "dataset_size": len(train_dataset),
        "eval_size": len(eval_dataset) if eval_dataset else 0,
        "training_time": datetime.now().isoformat(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if torch.cuda.is_available()
        else 0,
        "log_file": log_path,
        "eval_metrics": eval_metrics,
        "use_qlora": model_config.load_in_4bit,
    }
    with open(
        os.path.join(training_config.output_dir, "training_info.json"), "w"
    ) as f:
        json.dump(training_info, f, indent=2)

    # Post-train quick eval
    run_eval(trainer.model, tokenizer, device, eval_prompts)

    logging.info("✅ LoRA adapter saved to: %s", training_config.output_dir)
    logging.info("📄 Debug log: %s", log_path)
    logging.info("🎉 Training completed at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
