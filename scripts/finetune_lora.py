#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for RTX 4060 Ti
Version: 1.1.1
Author: Auto-generated from INSTRUCTIONS.md
Optimized for: RTX 4060 Ti (16GB VRAM)

Changelog:
Version: 1.2.0
- bf16 por defecto (más estable en Ada/Lovelace)
- Packing forzado (opcional por .env)
- Gradient Checkpointing + use_cache=False
- EarlyStopping + eval/save por pasos
- Opción QLoRA 4-bit activable por .env sin tocar el código
- v1.1.1: Expose training hyperparameters as module-level constants, add debug logging & post-train eval
- v1.1.0: Faster training loop (less repetition, bigger batches, fewer epochs, LoRA r=16)
- v1.0.9: Repeat tiny datasets and widen LoRA target modules for better learning
- v1.0.8: Tune hyperparameters for tiny datasets (more epochs, smaller batches)
- v1.0.7: Use dataset_text_field pipeline to avoid tokenizer batch errors
"""

import os
import sys
import math
import json
import logging
from typing import List
from datetime import datetime

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Opcional: BitsAndBytes solo si QLoRA activado y lib disponible
try:
    from transformers import BitsAndBytesConfig  # type: ignore
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

SCRIPT_VERSION = "1.2.0"
SCRIPT_NAME = "finetune_lora.py"

# -------- Helpers ENV --------
def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    except Exception as exc:
        print(f"⚠️ Could not load {path}: {exc}", file=sys.stderr)

def env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None: return default
    try: return int(v)
    except ValueError: return default

def env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None: return default
    try: return float(v)
    except ValueError: return default

def env_str(key: str, default: str) -> str:
    return os.getenv(key, default)

def env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None: return default
    return v.lower() in ("1", "true", "yes", "y", "on")

def env_list(key: str, default: List[str]) -> List[str]:
    v = os.getenv(key)
    if not v: return default
    items = [s.strip() for s in v.split(",") if s.strip()]
    return items or default

load_env_file()

# ======== Tunables por .env ========
MODEL_ID = env_str("FT_MODEL_ID", "microsoft/DialoGPT-medium")
DATA_PATH = env_str("FT_DATA_PATH", "data/instructions.jsonl")
OUT_DIR = env_str("FT_OUT_DIR", "models/out-lora")

DATASET_MIN_EXAMPLES = env_int("FT_DATASET_MIN_EXAMPLES", 240)
PER_DEVICE_BATCH_SIZE = env_int("FT_PER_DEVICE_BATCH_SIZE", 2)
GRADIENT_ACCUMULATION = env_int("FT_GRADIENT_ACCUMULATION", 8)
MAX_SEQ_LEN_OVERRIDE = env_int("FT_MAX_SEQ_LEN", 1024)

NUM_EPOCHS = env_int("FT_NUM_EPOCHS", 8)
LEARNING_RATE = env_float("FT_LEARNING_RATE", 1e-4)
WARMUP_RATIO = env_float("FT_WARMUP_RATIO", 0.15)
LR_SCHEDULER = env_str("FT_LR_SCHEDULER", "cosine")
WEIGHT_DECAY = env_float("FT_WEIGHT_DECAY", 0.01)

LORA_RANK = env_int("FT_LORA_RANK", 16)
LORA_ALPHA = env_int("FT_LORA_ALPHA", 32)
LORA_DROPOUT = env_float("FT_LORA_DROPOUT", 0.15)
LORA_TARGET_MODULES = env_list("FT_LORA_TARGET_MODULES", ["c_attn", "c_proj", "c_fc"])

LOGGING_STEPS = env_int("FT_LOGGING_STEPS", 10)
SAVE_STRATEGY = env_str("FT_SAVE_STRATEGY", "steps")
SAVE_TOTAL_LIMIT = env_int("FT_SAVE_TOTAL_LIMIT", 2)
DATASET_SHUFFLE_SEED = env_int("FT_DATASET_SHUFFLE_SEED", 42)
VALIDATION_SPLIT = env_float("FT_VALIDATION_SPLIT", 0.2)
DEBUG_LOG_FILE = env_str("FT_DEBUG_LOG_FILE", "debug_last_run.log")
EVAL_MAX_NEW_TOKENS = env_int("FT_EVAL_MAX_NEW_TOKENS", 220)
EVAL_SAMPLE_SIZE = env_int("FT_EVAL_SAMPLE_SIZE", 10)
EVAL_STEPS = env_int("FT_EVAL_STEPS", 100)
SAVE_STEPS = env_int("FT_SAVE_STEPS", 100)
FORCE_PACKING = env_bool("FT_FORCE_PACKING", True)
USE_QLORA = env_bool("FT_USE_QLORA", False)
TRUST_REMOTE_CODE = env_bool("FT_TRUST_REMOTE_CODE", False)

# Hardening/ruido
os.environ.setdefault("TOKENIZERS_PARALLELISM", os.getenv("TOKENIZERS_PARALLELISM", "false"))
os.environ.setdefault("HF_DATASETS_DISABLE_MULTIPROCESSING", os.getenv("HF_DATASETS_DISABLE_MULTIPROCESSING", "1"))

EVAL_FALLBACK_PROMPTS = [
    {"system": "Eres un asistente experto en procesos internos.",
     "user": "Dame los pasos para conciliar pagos de los lunes.",
     "expected": "1) Exporta el CSV del banco..."}
]

# -------- Logging --------
def setup_logging():
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", DEBUG_LOG_FILE)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Debug log initialized at %s", log_path)
    return log_path

def log_version_info():
    logging.info("🚀 %s v%s", SCRIPT_NAME, SCRIPT_VERSION)
    logging.info("📅 Started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("💻 PyTorch: %s", torch.__version__)
    dev = "CUDA" if torch.cuda.is_available() else "CPU"
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logging.info("🖥️ Device: %s", name)
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info("📊 VRAM: %.1fGB", vram_gb)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Datos --------
def format_example(example, eos_token: str):
    system_prompt = example.get("system", "Eres un asistente útil y conciso.")
    text = (
        f"System: {system_prompt}\n"
        f"User: {example['input']}\n"
        f"Assistant: {example['output']}{eos_token}"
    )
    return {"text": text}

def run_eval(model, tok, device, eval_prompts):
    if not eval_prompts:
        logging.warning(">> No evaluation prompts available. Skipping quick evaluation.")
        return
    logging.info(">> Running quick evaluation on %d sample prompts...", len(eval_prompts))
    model.eval()
    for sample in eval_prompts:
        composed = f"System: {sample.get('system','')}\nUser: {sample.get('user','')}\nAssistant:"
        inputs = tok(composed, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                repetition_penalty=1.05,
            )
        decoded = tok.decode(out[0], skip_special_tokens=True)
        reply = decoded.split("Assistant:")[-1].strip()
        logging.info("[Eval] User: %s", sample.get("user",""))
        logging.info("[Eval] Assistant: %s", reply)
        exp = sample.get("expected")
        if exp:
            logging.info("[Eval] Expected: %s", exp)

def main():
    log_path = setup_logging()
    log_version_info()

    # SDP kernels (Flash/Mem-efficient) si están disponibles
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    device = get_device()
    logging.info(">> Device detectado: %s", device)

    if not os.path.exists(DATA_PATH):
        logging.error("❌ Error: dataset %s not found.", DATA_PATH)
        return

    logging.info(">> Loading dataset from: %s", DATA_PATH)
    raw_ds = load_dataset("json", data_files=DATA_PATH)["train"]
    logging.info(">> Dataset loaded: %d examples", len(raw_ds))
    if len(raw_ds) < 2:
        logging.error("❌ Dataset too small (%d). Add more data before training.", len(raw_ds))
        return

    split = raw_ds.train_test_split(test_size=VALIDATION_SPLIT, seed=DATASET_SHUFFLE_SEED)
    train_raw = split["train"].shuffle(seed=DATASET_SHUFFLE_SEED)
    eval_raw = split["test"].shuffle(seed=DATASET_SHUFFLE_SEED)

    # Eval prompts
    eval_prompts = []
    sample_n = min(EVAL_SAMPLE_SIZE, len(eval_raw))
    for i in range(sample_n):
        row = eval_raw[i]
        eval_prompts.append(
            {"system": row.get("system", ""), "user": row.get("input", ""), "expected": row.get("output", "")}
        )
    if not eval_prompts:
        eval_prompts = EVAL_FALLBACK_PROMPTS

    logging.info(">> Loading tokenizer: %s", MODEL_ID)
    tokenizer_kwargs = {"trust_remote_code": True} if TRUST_REMOTE_CODE else {}
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID, **tokenizer_kwargs)
    except ValueError as exc:
        if "Tokenizer class" in str(exc) and not TRUST_REMOTE_CODE:
            logging.error("❌ Tokenizer for %s requires remote code. Set FT_TRUST_REMOTE_CODE=1 in your .env to allow it.", MODEL_ID)
            return
        raise
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    EOS = tok.eos_token
    logging.info(">> Tokenizer loaded")

    # Carga de modelo (bf16 o QLoRA 4-bit)
    model_kwargs = {}
    if USE_QLORA:
        if not _HAS_BNB:
            logging.error("❌ FT_USE_QLORA=true pero bitsandbytes no está instalado. `pip install bitsandbytes`")
            return
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_cfg
        logging.info(">> QLoRA 4-bit activado (NF4, compute bf16)")
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    logging.info(">> Loading model: %s", MODEL_ID)
    model_load_kwargs = dict(device_map="auto", **model_kwargs)
    if TRUST_REMOTE_CODE:
        model_load_kwargs["trust_remote_code"] = True
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            **model_load_kwargs,
        )
    except ValueError as exc:
        if "requires you to execute the modeling code" in str(exc) and not TRUST_REMOTE_CODE:
            logging.error("❌ Model %s requires remote code. Set FT_TRUST_REMOTE_CODE=1 in your .env to allow it.", MODEL_ID)
            return
        raise
    model.to(device)
    logging.info(">> Model loaded to device")

    # Longitud de contexto & packing
    # Selecciona el mínimo válido entre modelo/tokenizer/override
    candidates = []
    for ctx in (getattr(model.config, "n_positions", None), getattr(tok, "model_max_length", None), MAX_SEQ_LEN_OVERRIDE):
        if ctx and ctx != float("inf"):
            candidates.append(int(ctx))
    max_seq_len = max(8, min(candidates) if candidates else 1024)
    use_packing = True if FORCE_PACKING else (len(train_raw) >= 40 and max_seq_len >= 2048)
    logging.info(">> Using max sequence length: %d | Packing: %s", max_seq_len, use_packing)

    # LoRA
    peft_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    logging.info(">> LoRA configuration set (r=%d, targets=%s)", LORA_RANK, ",".join(LORA_TARGET_MODULES))

    # Checkpointing para subir batch y estabilidad
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
        logging.info(">> Gradient checkpointing enabled")
    except Exception as e:
        logging.warning("No se pudo habilitar gradient checkpointing: %s", e)

    # Dataset -> texto formateado + EOS
    logging.info(">> Formatting examples...")
    train_ds = train_raw.map(lambda ex: format_example(ex, EOS), remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(lambda ex: format_example(ex, EOS), remove_columns=eval_raw.column_names)

    # Expandir para pasos suficientes
    if len(train_ds) < DATASET_MIN_EXAMPLES:
        repeat_factor = max(1, math.ceil(DATASET_MIN_EXAMPLES / len(train_ds)))
        train_ds = concatenate_datasets([train_ds] * repeat_factor).shuffle(seed=DATASET_SHUFFLE_SEED)
        logging.info(">> Training dataset expanded via repetition: %dx -> %d samples", repeat_factor, len(train_ds))

    logging.info(">> Train samples: %d | Eval samples: %d", len(train_ds), len(eval_ds))

    # Args de entrenamiento
    sft_args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        weight_decay=WEIGHT_DECAY,
        dataloader_pin_memory=True,
        report_to="tensorboard",
        logging_dir="logs",
        bf16=not USE_QLORA,  # en QLoRA compute ya es bf16 vía bnb_cfg
        fp16=False,
        dataloader_num_workers=2,
        tf32=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch",
        save_safetensors=True,
    )
    logging.info(">> Training configuration set")

    # Trainer + EarlyStopping
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
        max_seq_length=max_seq_len,
        packing=use_packing,
        dataset_text_field="text",
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    logging.info(">> Trainer initialized")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logging.info(">> Starting training...")
    logging.info("   - Batch size: %s", sft_args.per_device_train_batch_size)
    logging.info("   - Grad accum: %s", sft_args.gradient_accumulation_steps)
    logging.info("   - Max seq len: %d", max_seq_len)
    logging.info("   - Epochs: %s", sft_args.num_train_epochs)
    logging.info("   - Train (after repeat): %d | Eval: %d", len(train_ds), len(eval_ds))

    trainer.train()

    eval_metrics = trainer.evaluate()
    logging.info(">> Validation metrics: %s", json.dumps(eval_metrics, indent=2))

    logging.info(">> Saving model (adapters + tokenizer)...")
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
        "use_qlora": USE_QLORA,
    }
    with open(os.path.join(OUT_DIR, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    # Post-train quick eval
    run_eval(trainer.model, tok, device, eval_prompts)

    logging.info("✅ Adaptador LoRA guardado en: %s", OUT_DIR)
    logging.info("📄 Debug log: %s", log_path)
    logging.info("🎉 Training completed at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()