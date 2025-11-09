#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for RTX 4060 Ti
Version: 1.1.1
Author: Auto-generated from INSTRUCTIONS.md
Optimized for: RTX 4060 Ti (16GB VRAM)

Changelog:
- v1.1.1: Expose training hyperparameters as module-level constants, add debug logging & post-train eval
- v1.1.0: Faster training loop (less repetition, bigger batches, fewer epochs, LoRA r=16)
- v1.0.9: Repeat tiny datasets and widen LoRA target modules for better learning
- v1.0.8: Tune hyperparameters for tiny datasets (more epochs, smaller batches)
- v1.0.7: Use dataset_text_field pipeline to avoid tokenizer batch errors
"""

import os
import sys
import math
import torch
import json
import logging
from typing import List
from datetime import datetime
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Version information
SCRIPT_VERSION = "1.1.1"
SCRIPT_NAME = "finetune_lora.py"


def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"\'")
                os.environ.setdefault(key, value)
    except Exception as exc:
        print(f"⚠️ Could not load {path}: {exc}", file=sys.stderr)


def env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def env_list(key: str, default: List[str]) -> List[str]:
    value = os.getenv(key)
    if value is None:
        return default
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or default


load_env_file()

# ======== Tunable Hyperparameters ========
MODEL_ID = env_str("FT_MODEL_ID", "microsoft/DialoGPT-medium")
DATA_PATH = env_str("FT_DATA_PATH", "data/instructions.jsonl")
OUT_DIR = env_str("FT_OUT_DIR", "models/out-tinyllama-lora")

# Dataset expansion & batching
DATASET_MIN_EXAMPLES = env_int("FT_DATASET_MIN_EXAMPLES", 240)
PER_DEVICE_BATCH_SIZE = env_int("FT_PER_DEVICE_BATCH_SIZE", 1)
GRADIENT_ACCUMULATION = env_int("FT_GRADIENT_ACCUMULATION", 12)

# Training schedule
NUM_EPOCHS = env_int("FT_NUM_EPOCHS", 12)
LEARNING_RATE = env_float("FT_LEARNING_RATE", 2e-5)
WARMUP_RATIO = env_float("FT_WARMUP_RATIO", 0.1)
LR_SCHEDULER = env_str("FT_LR_SCHEDULER", "linear")
WEIGHT_DECAY = env_float("FT_WEIGHT_DECAY", 0.1)

# LoRA configuration
LORA_RANK = env_int("FT_LORA_RANK", 32)
LORA_ALPHA = env_int("FT_LORA_ALPHA", 32)
LORA_DROPOUT = env_float("FT_LORA_DROPOUT", 0.3)
LORA_TARGET_MODULES = env_list("FT_LORA_TARGET_MODULES", ["c_attn", "c_proj"])

# Misc
LOGGING_STEPS = env_int("FT_LOGGING_STEPS", 5)
SAVE_STRATEGY = env_str("FT_SAVE_STRATEGY", "epoch")
SAVE_TOTAL_LIMIT = env_int("FT_SAVE_TOTAL_LIMIT", 3)
DATASET_SHUFFLE_SEED = env_int("FT_DATASET_SHUFFLE_SEED", 42)
VALIDATION_SPLIT = env_float("FT_VALIDATION_SPLIT", 0.2)
DEBUG_LOG_FILE = env_str("FT_DEBUG_LOG_FILE", "debug_last_run.log")
EVAL_MAX_NEW_TOKENS = env_int("FT_EVAL_MAX_NEW_TOKENS", 220)
EVAL_SAMPLE_SIZE = env_int("FT_EVAL_SAMPLE_SIZE", 10)
EVAL_FALLBACK_PROMPTS = [
    {
        "system": "Eres un asistente experto en procesos internos.",
        "user": "Dame los pasos para conciliar pagos de los lunes.",
        "expected": "1) Exporta el CSV del banco..."
    }
]


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", DEBUG_LOG_FILE)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Debug log initialized at %s", log_path)
    return log_path


def log_version_info():
    logging.info("🚀 %s v%s", SCRIPT_NAME, SCRIPT_VERSION)
    logging.info("📅 Started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("💻 PyTorch: %s", torch.__version__)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logging.info("🖥️ Device: %s", device_name)
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info("📊 VRAM: %.1fGB", vram_gb)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_example(example):
    system_prompt = example.get("system", "Eres un asistente útil y conciso.")
    prompt = (
        f"System: {system_prompt}\n"
        f"User: {example['input']}\n"
        f"Assistant: {example['output']}"
    )
    return {"text": prompt}


def run_eval(model, tok, device, eval_prompts):
    if not eval_prompts:
        logging.warning(">> No evaluation prompts available. Skipping quick evaluation.")
        return
    logging.info(">> Running quick evaluation on %d sample prompts...", len(eval_prompts))
    model.eval()
    for sample in eval_prompts:
        composed_prompt = (
            f"System: {sample['system']}\n"
            f"User: {sample['user']}\n"
            "Assistant:"
        )
        inputs = tok(composed_prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                repetition_penalty=1.05,
            )
        decoded = tok.decode(outputs[0], skip_special_tokens=True)
        assistant_reply = decoded.split("Assistant:")[-1].strip()
        logging.info("[Eval] User: %s", sample["user"])
        logging.info("[Eval] Assistant: %s", assistant_reply)
        expected = sample.get("expected")
        if expected:
            logging.info("[Eval] Expected: %s", expected)


def main():
    log_path = setup_logging()
    log_version_info()

    device = get_device()
    logging.info(">> Device detectado: %s", device)
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(">> GPU: %s", torch.cuda.get_device_name(0))
        logging.info(">> VRAM: %.1fGB", vram_gb)

    dataset_path = DATA_PATH

    if not os.path.exists(dataset_path):
        logging.error("❌ Error: dataset %s not found.", dataset_path)
        return

    logging.info(">> Loading dataset from: %s", dataset_path)
    raw_ds = load_dataset("json", data_files=dataset_path)["train"]
    logging.info(">> Dataset loaded: %d examples", len(raw_ds))

    if len(raw_ds) < 2:
        logging.error("❌ Dataset too small (%d samples). Add more data before training.", len(raw_ds))
        return

    split = raw_ds.train_test_split(test_size=VALIDATION_SPLIT, seed=DATASET_SHUFFLE_SEED)
    train_raw = split["train"].shuffle(seed=DATASET_SHUFFLE_SEED)
    eval_raw = split["test"].shuffle(seed=DATASET_SHUFFLE_SEED)

    eval_prompts = []
    sample_size = min(EVAL_SAMPLE_SIZE, len(eval_raw))
    if sample_size > 0:
        for idx in range(sample_size):
            row = eval_raw[idx]
            eval_prompts.append(
                {
                    "system": row.get("system", ""),
                    "user": row.get("input", ""),
                    "expected": row.get("output", ""),
                }
            )
    if not eval_prompts:
        eval_prompts = EVAL_FALLBACK_PROMPTS

    logging.info(">> Loading model: %s", MODEL_ID)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    logging.info(">> Tokenizer loaded")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.to(device)
    logging.info(">> Model loaded to device")

    model_context = getattr(model.config, "n_positions", None)
    tokenizer_context = getattr(tok, "model_max_length", None)
    default_context = 1024
    candidate_contexts = [default_context]
    for ctx in (model_context, tokenizer_context):
        if ctx is not None and ctx > 0 and ctx != float("inf"):
            candidate_contexts.append(int(ctx))
    max_seq_len = min(candidate_contexts)
    logging.info(">> Using max sequence length: %d", max_seq_len)

    peft_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    logging.info(">> LoRA configuration set (r=%d)", LORA_RANK)

    model = get_peft_model(model, peft_cfg)
    logging.info(">> LoRA model ready")

    logging.info(">> Formatting examples...")
    train_ds = train_raw.map(format_example, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(format_example, remove_columns=eval_raw.column_names)

    if len(train_ds) < DATASET_MIN_EXAMPLES:
        repeat_factor = max(1, math.ceil(DATASET_MIN_EXAMPLES / len(train_ds)))
        train_ds = concatenate_datasets([train_ds] * repeat_factor).shuffle(seed=DATASET_SHUFFLE_SEED)
        logging.info(">> Training dataset expanded via repetition: %dx -> %d samples", repeat_factor, len(train_ds))

    logging.info(">> Train samples: %d | Eval samples: %d", len(train_ds), len(eval_ds))

    sft_args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="epoch",
        save_strategy=SAVE_STRATEGY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        weight_decay=WEIGHT_DECAY,
        dataloader_pin_memory=True,
        report_to="tensorboard",
        logging_dir="logs",
        fp16=True,
        dataloader_num_workers=2,
        tf32=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch",
    )
    logging.info(">> Training configuration set")

    can_pack = len(train_ds) >= 40 and max_seq_len >= 2048
    use_packing = bool(can_pack)
    if use_packing:
        logging.info(">> Dataset has %d examples - enabling packing for better efficiency", len(train_ds))
    else:
        logging.info(">> Packing disabled (dataset=%d, max_seq_len=%d)", len(train_ds), max_seq_len)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
        max_seq_length=max_seq_len,
        packing=use_packing,
        dataset_text_field="text",
    )
    logging.info(">> Trainer initialized")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logging.info(">> Starting training...")
    logging.info("   - Batch size: %s", sft_args.per_device_train_batch_size)
    logging.info("   - Gradient accumulation: %s", sft_args.gradient_accumulation_steps)
    logging.info("   - Max sequence length: %d", max_seq_len)
    logging.info("   - Training epochs: %s", sft_args.num_train_epochs)
    logging.info("   - Total training examples (after repeat): %d", len(train_ds))
    logging.info("   - Total evaluation examples: %d", len(eval_ds))

    trainer.train()

    eval_metrics = trainer.evaluate()
    logging.info(">> Validation metrics: %s", json.dumps(eval_metrics, indent=2))

    logging.info(">> Saving model...")
    trainer.model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

    training_info = {
        "script_version": SCRIPT_VERSION,
        "model_id": MODEL_ID,
        "dataset_size": len(train_ds),
        "eval_size": len(eval_ds),
        "training_time": datetime.now().isoformat(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
        "log_file": os.path.join("logs", DEBUG_LOG_FILE),
        "eval_metrics": eval_metrics,
    }

    with open(os.path.join(OUT_DIR, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    logging.info("✅ Adaptador LoRA guardado en: %s", OUT_DIR)
    logging.info("✅ Logs disponibles en: logs/")
    logging.info("✅ Training info saved to training_info.json")

    run_eval(trainer.model, tok, device, eval_prompts)

    logging.info("🎉 Training completed at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("📄 Debug log stored at %s", log_path)


if __name__ == "__main__":
    main()
