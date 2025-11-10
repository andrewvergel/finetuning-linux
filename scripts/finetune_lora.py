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

import torch
from datasets import Dataset, concatenate_datasets
from packaging import version
from transformers import TrainingArguments, EarlyStoppingCallback
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
        ">> Training dataset expanded: %dx -> %d ejemplos", repeat_factor, len(expanded)
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
            "Packing forzado deshabilitado: dataset muy pequeño (%d < %d ejemplos)",
            len(dataset),
            min_examples,
        )
        return False

    if stats["approx_total_tokens"] < min_tokens_for_packing:
        logging.warning(
            "Packing forzado deshabilitado: tokens estimados insuficientes (%.0f < %d)",
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
    logging.info(">> Device detectado: %s", device)

    # Validate dataset exists
    if not Path(data_config.train_path).exists():
        logging.error("❌ Error: dataset %s not found.", data_config.train_path)
        return

    logging.info(">> Loading dataset from: %s", data_config.train_path)

    # Load model and tokenizer using ModelBuilder
    logging.info(">> Loading model: %s", model_config.model_name_or_path)
    model_builder = ModelBuilder(model_config=model_config, training_config=training_config)
    model, tokenizer = model_builder.load_model()

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
    dataset_min_examples = int(os.getenv("FT_DATASET_MIN_EXAMPLES", "100"))
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
    logging.info("📊 Estadísticas del dataset:")
    logging.info(f"  - Ejemplos: {stats['total_examples']}")
    logging.info(
        f"  - Tokens/ejemplo: {stats['avg_tokens']:.1f} "
        f"(min: {stats['min_tokens']}, max: {stats['max_tokens']})"
    )
    logging.info(f"  - Caracteres/ejemplo: {stats['avg_chars']:.1f}")
    logging.info(f"  - Tokens totales estimados: {stats['approx_total_tokens']:,.0f}")

    # Determine if packing should be used
    force_packing = bool(os.getenv("FT_FORCE_PACKING", "false").lower() in ("true", "1", "yes"))
    use_packing = should_use_packing(train_dataset, stats, max_seq_len, force_packing)
    logging.info(f"  - Packing: {'HABILITADO' if use_packing else 'DESHABILITADO'}")

    if stats["max_tokens"] > max_seq_len * 0.9:
        logging.warning(
            "¡Atención! Algunos ejemplos están cerca del límite de contexto (%d tokens)",
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

    # Initialize SFTTrainer
    logging.info("🚀 Inicializando SFTTrainer...")

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

    trainer = SFTTrainer(**sft_trainer_kwargs)

    # Verify model is ready for training
    logging.info(f">> Model device: {next(trainer.model.parameters()).device}")
    logging.info(f">> Model training mode: {trainer.model.training}")

    # Create output directory
    os.makedirs(training_config.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Log configuration summary
    logging.info("\n" + "=" * 80)
    logging.info("🚀 INICIANDO ENTRENAMIENTO")
    logging.info("=" * 80)
    logging.info("📋 Configuración:")
    logging.info(f"   - Modelo: {model_config.model_name_or_path}")
    logging.info(
        f"   - Batch size: {training_args.per_device_train_batch_size} "
        f"(x{training_args.gradient_accumulation_steps})"
    )
    logging.info(f"   - Longitud máxima: {max_seq_len} tokens")
    logging.info(f"   - Épocas: {training_args.num_train_epochs}")
    logging.info(
        f"   - Tamaño del dataset: {len(train_dataset)} entrenamiento, "
        f"{len(eval_dataset) if eval_dataset else 0} validación"
    )
    logging.info(f"   - Learning rate: {training_args.learning_rate}")
    logging.info(f"   - Peso de decaimiento: {training_args.weight_decay}")
    logging.info(
        f"   - LoRA r={model_config.lora_rank}, alpha={model_config.lora_alpha}, "
        f"dropout={model_config.lora_dropout}"
    )
    logging.info(f"   - QLoRA: {'Sí' if model_config.load_in_4bit else 'No'}")
    logging.info(f"   - Packing: {'Sí' if use_packing else 'No'}")
    logging.info("=" * 80 + "\n")

    # Training with error handling
    try:
        logging.info("🏋️ Iniciando entrenamiento...")
        train_result = trainer.train()

        # Save training metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        # Final evaluation
        eval_metrics = {}
        if eval_dataset and len(eval_dataset) > 0:
            logging.info("📊 Evaluando modelo final...")
            eval_metrics = trainer.evaluate()
            metrics.update({"eval_" + k: v for k, v in eval_metrics.items()})

        # Save metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        logging.info("✅ Entrenamiento completado con éxito!")
        logging.info("📊 Métricas de evaluación:")
        for k, v in eval_metrics.items():
            logging.info(f"   - {k}: {v:.4f}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logging.error(
                "❌ Error: Memoria insuficiente. Intenta reducir el tamaño de batch o secuencia."
            )
        raise
    except Exception as e:
        logging.error(f"❌ Error durante el entrenamiento: {str(e)}")
        # Save model despite error if possible
        try:
            trainer.save_model(training_config.output_dir + "_crashed")
            logging.info(f"Modelo guardado en {training_config.output_dir}_crashed")
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

    logging.info("✅ Adaptador LoRA guardado en: %s", training_config.output_dir)
    logging.info("📄 Debug log: %s", log_path)
    logging.info("🎉 Training completed at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
