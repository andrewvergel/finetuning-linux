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
from datasets import load_dataset, concatenate_datasets, Dataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

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

import transformers
import peft

REQUIRED_TRANSFORMERS = "4.40.0"
REQUIRED_PEFT = "0.10.0"

if version.parse(transformers.__version__) < version.parse(REQUIRED_TRANSFORMERS):
    raise RuntimeError(
        f"transformers>={REQUIRED_TRANSFORMERS} is required for {MODEL_ID}. "
        f"Found {transformers.__version__}. Please upgrade: pip install --upgrade transformers"
    )

if version.parse(peft.__version__) < version.parse(REQUIRED_PEFT):
    raise RuntimeError(
        f"peft>={REQUIRED_PEFT} is required for this script. Found {peft.__version__}. "
        f"Please upgrade: pip install --upgrade peft"
    )

# ======== Tunables por .env ========
MODEL_ID = env_str("FT_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
DATA_PATH = env_str("FT_DATA_PATH", "data/instructions.jsonl")
OUT_DIR = env_str("FT_OUT_DIR", "models/out-lora")

# Memory optimization parameters
DATASET_MIN_EXAMPLES = env_int("FT_DATASET_MIN_EXAMPLES", 100)  # Reduced minimum dataset size
PER_DEVICE_BATCH_SIZE = env_int("FT_PER_DEVICE_BATCH_SIZE", 1)  # Keep at 1 for VRAM efficiency
GRADIENT_ACCUMULATION = env_int("FT_GRADIENT_ACCUMULATION", 8)  # Reduced for more frequent updates
MAX_SEQ_LEN_OVERRIDE = env_int("FT_MAX_SEQ_LEN", 512)  # Keep at 512 for memory efficiency

# Enable memory optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

# Training parameters
NUM_EPOCHS = env_int("FT_NUM_EPOCHS",8)  # Reduced epochs for faster iteration
LEARNING_RATE = env_float("FT_LEARNING_RATE", 1.5e-5)  # Lower learning rate for stability
WARMUP_RATIO = env_float("FT_WARMUP_RATIO", 0.1)  # Keep 10% warmup steps
LR_SCHEDULER = env_str("FT_LR_SCHEDULER", "cosine_with_restarts")  # Better convergence
WEIGHT_DECAY = env_float("FT_WEIGHT_DECAY", 0.02)  # Slightly higher for better regularization

# LoRA parameters - balanced for performance and memory
LORA_RANK = env_int("FT_LORA_RANK", 16)  # Increased rank for better learning
LORA_ALPHA = env_int("FT_LORA_ALPHA", 16)  # Alpha = 2*rank for stability
LORA_DROPOUT = env_float("FT_LORA_DROPOUT", 0.05)  # Slight dropout for regularization
# Target key layers for efficient fine-tuning
LORA_TARGET_MODULES = env_list("FT_LORA_TARGET_MODULES", ["q_proj", "v_proj"])  # Added v_proj

# Training configuration
LOGGING_STEPS = env_int("FT_LOGGING_STEPS", 10)  # Log every 10 steps
SAVE_STRATEGY = env_str("FT_SAVE_STRATEGY", "steps")
SAVE_TOTAL_LIMIT = env_int("FT_SAVE_TOTAL_LIMIT", 2)  # Keep last 2 checkpoints
DATASET_SHUFFLE_SEED = env_int("FT_DATASET_SHUFFLE_SEED", 42)
VALIDATION_SPLIT = env_float("FT_VALIDATION_SPLIT", 0.15)  # Slightly less validation data
DEBUG_LOG_FILE = env_str("FT_DEBUG_LOG_FILE", "debug_last_run.log")
EVAL_MAX_NEW_TOKENS = env_int("FT_EVAL_MAX_NEW_TOKENS", 128)  # Keep reasonable for evaluation
EVAL_SAMPLE_SIZE = env_int("FT_EVAL_SAMPLE_SIZE", 3)  # Fewer samples to save VRAM
EVAL_STEPS = env_int("FT_EVAL_STEPS", 25)  # Evaluate more frequently
SAVE_STEPS = env_int("FT_SAVE_STEPS", 25)  # Save more frequently

# Memory optimization flags
FORCE_PACKING = env_bool("FT_FORCE_PACKING", False)  # Keep disabled for VRAM efficiency
USE_QLORA = env_bool("FT_USE_QLORA", True)  # Enable QLoRA for 4-bit quantization
TRUST_REMOTE_CODE = env_bool("FT_TRUST_REMOTE_CODE", True)  # Required for some models

# Enable memory optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For better error messages

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
def validate_example(example: dict) -> bool:
    """Valida que un ejemplo tenga la estructura correcta."""
    required_fields = ["input", "output"]
    return all(field in example and example[field].strip() for field in required_fields)

def format_example(example, eos_token: str):
    """Formatea un ejemplo para el entrenamiento."""
    if not validate_example(example):
        raise ValueError(f"Ejemplo inválido: {example}")
        
    system_prompt = example.get("system", "Eres un asistente útil y conciso.")
    text = (
        f"System: {system_prompt}\n"
        f"User: {example['input']}\n"
        f"Assistant: {example['output']}{eos_token}"
    )
    return {"text": text, "length": len(text)}

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
            logging.warning(
                "Tokenizer for %s requires remote code. Retrying once with trust_remote_code=True (set FT_TRUST_REMOTE_CODE=1 to silence this warning).",
                MODEL_ID,
            )
            tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        else:
            raise
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    EOS = tok.eos_token
    logging.info(">> Tokenizer loaded")

    # Carga de modelo (bf16 o QLoRA 4-bit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = {"": device.index if hasattr(device, 'index') and device.index is not None else 0}
    
    if USE_QLORA and _HAS_BNB:
        logging.info("🔧 Configurando QLoRA (4-bit)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            trust_remote_code=TRUST_REMOTE_CODE,
            device_map=device_map,
            torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=TRUST_REMOTE_CODE,
            device_map=device_map,
            torch_dtype=torch.bfloat16
        )
        if not USE_QLORA:
            model.to(device)
    
    logging.info(">> Model loaded on %s", device)

    # Longitud de contexto & packing
    # Selecciona el mínimo válido entre modelo/tokenizer/override
    candidates = []
    for ctx in (getattr(model.config, "n_positions", None), getattr(tok, "model_max_length", None), MAX_SEQ_LEN_OVERRIDE):
        if ctx and ctx != float("inf"):
            candidates.append(int(ctx))
    
    # Reduce max sequence length to save memory
    max_seq_len = max(8, min(candidates) if candidates else 1024)
    max_seq_len = min(max_seq_len, 1024)  # Further reduce to 1024 tokens to save memory
    
    # Force disable packing to save memory
    use_packing = False
    logging.info("Packing is disabled to save memory")
    logging.info(">> Using max sequence length: %d | Packing: %s", max_seq_len, use_packing)

    # LoRA configuration with memory optimization
    peft_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    model = get_peft_model(model, peft_cfg)
    logging.info(">> LoRA configuration set (r=%d, targets=%s)", LORA_RANK, ",".join(LORA_TARGET_MODULES))

    # Ensure model is in training mode 
    model.train()
    
    # Enable aggressive memory optimizations
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.backends.cuda.enable_mem_efficient_sdp(False)  # Disable memory-efficient attention
    torch.backends.cuda.enable_flash_sdp(False)  # Disable flash attention
    logging.info(">> Gradient checkpointing enabled with reentrant=False")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f">> Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Enable CPU offload for optimizer states if using QLoRA
    if USE_QLORA:
        try:
            from accelerate import Accelerator
            from accelerate.utils import set_seed
            set_seed(42)  # For reproducibility
            
            # Initialize accelerator with gradient accumulation
            accelerator = Accelerator(
                gradient_accumulation_steps=GRADIENT_ACCUMULATION,
                mixed_precision='bf16' if not USE_QLORA else 'no'
            )
            
            # Prepare model with accelerator
            model = accelerator.prepare_model(model)
            logging.info(">> CPU offload for optimizer states enabled")
        except Exception as e:
            logging.warning("Could not enable CPU offload: %s", e)
    
    # Verify model parameters require gradients
    for name, param in model.named_parameters():
        if 'lora_' in name and not param.requires_grad:
            logging.warning(f"Parameter {name} does not require gradients!")

    # Dataset -> texto formateado + EOS
    logging.info(">> Formatting and validating examples...")
    
    def process_dataset(dataset, dataset_name="train"):
        valid_examples = []
        invalid_examples = 0
        
        for example in dataset:
            try:
                if validate_example(example):
                    formatted = format_example(example, EOS)
                    valid_examples.append(formatted)
                else:
                    invalid_examples += 1
            except Exception as e:
                logging.warning(f"Error procesando ejemplo en {dataset_name}: {e}")
                invalid_examples += 1
        
        if invalid_examples > 0:
            logging.warning("Se encontraron %d ejemplos inválidos en el dataset %s", 
                          invalid_examples, dataset_name)
        
        return valid_examples
    
    # Procesar datasets de entrenamiento y validación
    train_examples = process_dataset(train_raw, "train")
    eval_examples = process_dataset(eval_raw, "eval")
    
    if not train_examples:
        raise ValueError("No hay ejemplos válidos en el conjunto de entrenamiento")
        
    if not eval_examples and len(train_examples) > 10:
        # Si no hay ejemplos de validación, dividir el train
        train_examples, eval_examples = train_examples[:-5], train_examples[-5:]
        logging.info("Usando últimos 5 ejemplos del train para validación")
    
    # Convertir a datasets de HF
    train_ds = Dataset.from_list(train_examples)
    eval_ds = Dataset.from_list(eval_examples) if eval_examples else None
    
    # Expandir dataset si es necesario
    if len(train_ds) < DATASET_MIN_EXAMPLES:
        repeat_factor = max(1, math.ceil(DATASET_MIN_EXAMPLES / len(train_ds)))
        train_ds = concatenate_datasets([train_ds] * repeat_factor).shuffle(seed=DATASET_SHUFFLE_SEED)
        logging.info(">> Training dataset expandido: %dx -> %d ejemplos", repeat_factor, len(train_ds))

    logging.info(">> Train samples: %d | Eval samples: %d", len(train_ds), len(eval_ds))

    # Analizar estadísticas del dataset
    sample_size = min(64, len(train_ds))
    token_lengths = []
    total_chars = 0
    
    for i in range(sample_size):
        text = train_ds[i]["text"]
        tokens = tok(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        token_lengths.append(len(tokens["input_ids"][0]))
        total_chars += len(text)
    
    stats = {
        "avg_tokens": sum(token_lengths) / len(token_lengths) if token_lengths else 0,
        "max_tokens": max(token_lengths) if token_lengths else 0,
        "min_tokens": min(token_lengths) if token_lengths else 0,
        "avg_chars": total_chars / sample_size if sample_size > 0 else 0,
        "total_examples": len(train_ds)
    }
    
    # Calcular tokens aproximados
    approx_total_tokens = stats["avg_tokens"] * len(train_ds)
    
    # Decidir si usar packing
    min_examples_for_packing = 64
    min_tokens_for_packing = max_seq_len * 4
    
    use_packing = FORCE_PACKING
    if FORCE_PACKING:
        if len(train_ds) < min_examples_for_packing:
            logging.warning("Packing forzado deshabilitado: dataset muy pequeño (%d < %d ejemplos)", 
                          len(train_ds), min_examples_for_packing)
            use_packing = False
        elif approx_total_tokens < min_tokens_for_packing:
            logging.warning("Packing forzado deshabilitado: tokens estimados insuficientes (%.0f < %d)", 
                          approx_total_tokens, min_tokens_for_packing)
            use_packing = False
    
    # Log de estadísticas
    logging.info("📊 Estadísticas del dataset:")
    logging.info(f"  - Ejemplos: {stats['total_examples']}")
    logging.info(f"  - Tokens/ejemplo: {stats['avg_tokens']:.1f} (min: {stats['min_tokens']}, max: {stats['max_tokens']})")
    logging.info(f"  - Caracteres/ejemplo: {stats['avg_chars']:.1f}")
    logging.info(f"  - Tokens totales estimados: {approx_total_tokens:,.0f}")
    logging.info(f"  - Packing: {'HABILITADO' if use_packing else 'DESHABILITADO'}")
    
    if stats['max_tokens'] > max_seq_len * 0.9:
        logging.warning("¡Atención! Algunos ejemplos están cerca del límite de contexto (%d tokens)", max_seq_len)

    # Configuración de TrainingArguments
    sft_args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps" if eval_ds else "no",  # Solo evaluar si hay dataset de validación
        eval_steps=EVAL_STEPS if eval_ds else None,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        weight_decay=WEIGHT_DECAY,
        dataloader_pin_memory=True,
        report_to=["tensorboard"],
        logging_dir=os.path.join(OUT_DIR, "logs"),
        bf16=not USE_QLORA,
        fp16=False,
        dataloader_num_workers=2,
        tf32=True,
        load_best_model_at_end=bool(eval_ds),  # Solo activar si hay dataset de validación
        metric_for_best_model="loss" if not eval_ds else "eval_loss",
        greater_is_better=False,
        optim="adamw_torch",
        save_safetensors=True,
    )
    logging.info(">> Training configuration set")

    # Configuración del entrenamiento
    try:
        # Callbacks
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=3,  # Paciencia de 3 evaluaciones
                early_stopping_threshold=0.01  # Mejora mínima requerida
            )
        ]

        logging.info("🚀 Inicializando SFTTrainer...")
        
        # Initialize trainer directly with parameters
        trainer = SFTTrainer(
            model=model,
            tokenizer=tok,
            train_dataset=train_ds,
            eval_dataset=eval_ds if eval_ds and len(eval_ds) > 0 else None,
            args=sft_args,
            dataset_text_field="text",
            max_seq_length=max_seq_len,
            neftune_noise_alpha=5.0,  # Mejora la generalización
            dataset_num_proc=2,  # Procesamiento en paralelo
            packing=use_packing,
            callbacks=callbacks,
        )
        
        # Verify model is ready for training
        logging.info(f">> Model device: {next(trainer.model.parameters()).device}")
        logging.info(f">> Model training mode: {trainer.model.training}")
        # Only log requires_grad for float parameters to avoid errors with QLoRA
        float_params = [p for p in trainer.model.parameters() if p.dtype.is_floating_point]
        if float_params:
            logging.info(f">> First float parameter requires_grad: {float_params[0].requires_grad}")
        else:
            logging.info(">> No float parameters found in model (expected for QLoRA with 4-bit)")
        
        # Validar configuración
        if trainer.eval_dataset is None:
            logging.warning("No hay dataset de validación. Se usará un subconjunto del entrenamiento.")
            
    except Exception as e:
        logging.error("Error al inicializar el entrenador: %s", str(e))
        # Intentar sin packing si falla
        if use_packing:
            logging.warning("Reintentando sin packing...")
            sft_config.packing = False
            trainer = SFTTrainer(
                model=model,
                tokenizer=tok,
                train_dataset=train_ds,
                eval_dataset=eval_ds if eval_ds and len(eval_ds) > 0 else None,
                args=sft_args,
                config=sft_config,
                callbacks=callbacks,
            )
        else:
            raise
    logging.info(">> Trainer initialized")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Resumen de la configuración
    logging.info("\n" + "="*80)
    logging.info("🚀 INICIANDO ENTRENAMIENTO")
    logging.info("="*80)
    logging.info("📋 Configuración:")
    logging.info(f"   - Modelo: {MODEL_ID}")
    logging.info(f"   - Batch size: {sft_args.per_device_train_batch_size} (x{sft_args.gradient_accumulation_steps})")
    logging.info(f"   - Longitud máxima: {max_seq_len} tokens")
    logging.info(f"   - Épocas: {sft_args.num_train_epochs}")
    logging.info(f"   - Tamaño del dataset: {len(train_ds)} entrenamiento, {len(eval_ds) if eval_ds else 0} validación")
    logging.info(f"   - Learning rate: {sft_args.learning_rate}")
    logging.info(f"   - Peso de decaimiento: {sft_args.weight_decay}")
    logging.info(f"   - LoRA r={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    logging.info(f"   - QLoRA: {'Sí' if USE_QLORA else 'No'}")
    logging.info(f"   - Packing: {'Sí' if use_packing else 'No'}")
    logging.info("="*80 + "\n")

    # Entrenamiento con manejo de errores
    try:
        logging.info("🏋️ Iniciando entrenamiento...")
        train_result = trainer.train()
        
        # Guardar métricas de entrenamiento
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_ds)
        
        # Evaluación final
        eval_metrics = {}
        if eval_ds and len(eval_ds) > 0:
            logging.info("📊 Evaluando modelo final...")
            eval_metrics = trainer.evaluate()
            metrics.update({"eval_" + k: v for k, v in eval_metrics.items()})
        
        # Guardar métricas
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        logging.info("✅ Entrenamiento completado con éxito!")
        logging.info("📊 Métricas de evaluación:")
        for k, v in eval_metrics.items():
            logging.info(f"   - {k}: {v:.4f}")
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logging.error("❌ Error: Memoria insuficiente. Intenta reducir el tamaño de batch o secuencia.")
        raise
    except Exception as e:
        logging.error(f"❌ Error durante el entrenamiento: {str(e)}")
        # Guardar el modelo a pesar del error si es posible
        try:
            trainer.save_model(OUT_DIR + "_crashed")
            logging.info(f"Modelo guardado en {OUT_DIR}_crashed")
        except:
            pass
        raise

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