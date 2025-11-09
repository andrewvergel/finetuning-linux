#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for RTX 4060 Ti
Version: 1.1.1
Author: Auto-generated from INSTRUCTIONS.md
Optimized for: RTX 4060 Ti (16GB VRAM)

Changelog:
- v1.1.1: Expose training hyperparameters as module-level constants for easy tuning
- v1.1.0: Faster training loop (less repetition, bigger batches, fewer epochs, LoRA r=16)
- v1.0.9: Repeat tiny datasets and widen LoRA target modules for better learning
- v1.0.8: Tune hyperparameters for tiny datasets (more epochs, smaller batches)
- v1.0.7: Use dataset_text_field pipeline to avoid tokenizer batch errors
"""

import os
import math
import torch
import json
from datetime import datetime
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Version information
SCRIPT_VERSION = "1.1.1"
SCRIPT_NAME = "finetune_lora.py"

# ======== Tunable Hyperparameters ========
MODEL_ID = "microsoft/DialoGPT-medium"
DATA_PATH = "data/instructions.jsonl"
OUT_DIR = "models/out-tinyllama-lora"

# Dataset expansion & batching
DATASET_MIN_EXAMPLES = 120
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 1

# Training schedule
NUM_EPOCHS = 15
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LR_SCHEDULER = "constant_with_warmup"
WEIGHT_DECAY = 0.0

# LoRA configuration
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["c_attn", "c_proj"]

# Misc
LOGGING_STEPS = 5
SAVE_STEPS = 100
DATASET_SHUFFLE_SEED = 42

def log_version_info():
    """Log script version and system information"""
    print(f"🚀 {SCRIPT_NAME} v{SCRIPT_VERSION}")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # System info
    print(f"💻 PyTorch: {torch.__version__}")
    print(f"🖥️ Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"📊 VRAM: {vram_gb:.1f}GB")

def get_device():
    """Auto-detect device available"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def main():
    """Main fine-tuning function"""
    # Version info
    log_version_info()
    
    # Configuration
    device = get_device()
    print(f">> Device detectado: {device}")
    if torch.cuda.is_available():
        print(f">> GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f">> VRAM: {vram_gb:.1f}GB")
    
    # Model and paths
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found!")
        print("Please create the dataset first using the instructions.")
        return
    
    print(f">> Loading dataset from: {DATA_PATH}")
    
    # Cargar dataset
    ds = load_dataset("json", data_files=DATA_PATH)["train"]
    print(f">> Dataset loaded: {len(ds)} examples")
    
    # Tokenizer + modelo base
    print(f">> Loading model: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(">> Tokenizer loaded")
    
    # Load model with optimization for RTX 4060 Ti
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,  # Use FP16 to save VRAM
        device_map="auto",          # Automatic distribution
    )
    model.to(device)
    print(">> Model loaded to device")
    
    # Determine safe max sequence length based on model/tokenizer limits
    model_context = getattr(model.config, "n_positions", None)
    tokenizer_context = getattr(tok, "model_max_length", None)
    default_context = 1024  # GPT-2 family default context size
    candidate_contexts = [default_context]
    for ctx in (model_context, tokenizer_context):
        if ctx is not None and ctx > 0 and ctx != float("inf"):
            candidate_contexts.append(int(ctx))
    max_seq_len = min(candidate_contexts)
    print(f">> Using max sequence length: {max_seq_len}")
    
    # Configuración LoRA optimizada para RTX 4060 Ti
    peft_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(">> LoRA configuration set")
    
    model = get_peft_model(model, peft_cfg)
    print(">> LoRA model ready")
    
    # Formateo de ejemplos
    def format_example(ex):
        sys = ex.get("system","Eres un asistente útil y conciso.")
        prompt = f"System: {sys}\nUser: {ex['input']}\nAssistant: {ex['output']}"
        return {"text": prompt}
    
    print(">> Formatting examples...")
    ds = ds.map(format_example, remove_columns=ds.column_names)

    if len(ds) < DATASET_MIN_EXAMPLES:
        repeat_factor = max(1, math.ceil(DATASET_MIN_EXAMPLES / len(ds)))
        ds = concatenate_datasets([ds] * repeat_factor).shuffle(seed=DATASET_SHUFFLE_SEED)
        print(f">> Dataset expanded via repetition: {repeat_factor}x -> {len(ds)} samples")
    
    # Configuración optimizada para RTX 4060 Ti (16GB VRAM)
    sft_args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        evaluation_strategy="no",
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        weight_decay=WEIGHT_DECAY,
        dataloader_pin_memory=True,
        report_to="tensorboard",
        logging_dir="logs",
        fp16=True,
        dataloader_num_workers=2,
        tf32=True,
    )
    print(">> Training configuration set")
    
    # Initialize trainer
    can_pack = len(ds) >= 40 and max_seq_len >= 2048
    use_packing = bool(can_pack)
    if use_packing:
        print(f">> Dataset has {len(ds)} examples - enabling packing for better efficiency")
    else:
        print(f">> Packing disabled (dataset={len(ds)}, max_seq_len={max_seq_len})")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        args=sft_args,
        max_seq_length=max_seq_len,
        packing=use_packing,
        dataset_text_field="text",
    )
    print(">> Trainer initialized")
    
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Entrenar modelo
    print(">> Starting training...")
    print(f"   - Batch size: {sft_args.per_device_train_batch_size}")
    print(f"   - Gradient accumulation: {sft_args.gradient_accumulation_steps}")
    print(f"   - Max sequence length: {max_seq_len}")
    print(f"   - Training epochs: {sft_args.num_train_epochs}")
    print(f"   - Total training examples (after repeat): {len(ds)}")
    
    trainer.train()
    
    # Save model
    print(">> Saving model...")
    trainer.model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    
    # Save training info
    training_info = {
        "script_version": SCRIPT_VERSION,
        "model_id": MODEL_ID,
        "dataset_size": len(ds),
        "training_time": datetime.now().isoformat(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0
    }
    
    with open(os.path.join(OUT_DIR, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("✅ Adaptador LoRA guardado en:", OUT_DIR)
    print("✅ Logs disponibles en:", "logs/")
    print("✅ Training info saved to training_info.json")
    print(f"🎉 Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
