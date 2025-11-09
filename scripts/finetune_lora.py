#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for RTX 4060 Ti
Version: 1.0.2
Author: Auto-generated from INSTRUCTIONS.md
Optimized for: RTX 4060 Ti (16GB VRAM)

Changelog:
- v1.0.2: Fixed packing for small datasets (disable packing if < 10 examples)
- v1.0.1: Fixed TRL compatibility for v0.7.4, added TF32 support
- v1.0.0: Initial version with RTX 4060 Ti optimization
"""

import os
import torch
import json
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Version information
SCRIPT_VERSION = "1.0.2"
SCRIPT_NAME = "finetune_lora.py"

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
    MODEL_ID = "microsoft/DialoGPT-medium"  # Optimized for RTX 4060 Ti
    DATA_PATH = "data/instructions.jsonl"
    OUT_DIR = "models/out-tinyllama-lora"
    
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
    
    # Configuración LoRA optimizada para RTX 4060 Ti
    peft_cfg = LoraConfig(
        r=32, lora_alpha=32, lora_dropout=0.05,
        target_modules=["c_attn"],  # Adjust according to model
        bias="none", task_type="CAUSAL_LM",
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
    
    # Configuración optimizada para RTX 4060 Ti (16GB VRAM)
    sft_args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=6,  # Adjusted for RTX 4060 Ti
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="no",  # Changed from eval_strategy
        warmup_ratio=0.03,
        dataloader_pin_memory=True,
        report_to="tensorboard",
        logging_dir="logs",
        fp16=True,  # Changed from bf16=False to fp16=True
        dataloader_num_workers=2,  # Optimized for RTX 4060 Ti
        tf32=True,  # Added TF32 support
        # Removed invalid parameters: max_seq_length, packing
    )
    print(">> Training configuration set")
    
    # Initialize trainer
    # Enable packing only if dataset has enough samples (minimum 10 for packing)
    use_packing = len(ds) >= 10
    if not use_packing:
        print(f">> Dataset has {len(ds)} examples - disabling packing for small dataset")
    else:
        print(f">> Dataset has {len(ds)} examples - enabling packing")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        args=sft_args,
        formatting_func=lambda ex: ex["text"],
        max_seq_length=2048,  # Add max_seq_length here instead of in TrainingArguments
        packing=use_packing,  # Conditionally enable packing based on dataset size
    )
    print(">> Trainer initialized")
    
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Entrenar modelo
    print(">> Starting training...")
    print(f"   - Batch size: {sft_args.per_device_train_batch_size}")
    print(f"   - Gradient accumulation: {sft_args.gradient_accumulation_steps}")
    print(f"   - Max sequence length: {sft_args.max_seq_length}")
    print(f"   - Training epochs: {sft_args.num_train_epochs}")
    
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
