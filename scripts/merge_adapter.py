#!/usr/bin/env python3
"""
LoRA Merge Adapter Script for RTX 4060 Ti
Version: 1.0.0
Author: Auto-generated from INSTRUCTIONS.md
Optimized for: RTX 4060 Ti (16GB VRAM)

Changelog:
- v1.0.0: Initial version with RTX 4060 Ti optimization
"""

import os
import json
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Version information
SCRIPT_VERSION = "1.0.0"
SCRIPT_NAME = "merge_adapter.py"

def log_version_info():
    """Log script version and system information"""
    print(f"🔗 {SCRIPT_NAME} v{SCRIPT_VERSION}")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # System info
    print(f"💻 PyTorch: {torch.__version__}")
    print(f"🖥️ Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"📊 VRAM: {vram_gb:.1f}GB")

def check_adapter_exists(adapter_path):
    """Check if the LoRA adapter exists"""
    if not os.path.exists(adapter_path):
        print(f"❌ Error: LoRA adapter not found at {adapter_path}")
        print("Please run finetune_lora.py first to create the adapter.")
        return False
    
    # Check for required files
    required_files = ["adapter_config.json", "adapter_model.bin"]
    for file in required_files:
        if not os.path.exists(os.path.join(adapter_path, file)):
            print(f"❌ Error: Required file {file} not found in {adapter_path}")
            return False
    
    print(f"✅ LoRA adapter found at {adapter_path}")
    return True

def load_training_info(adapter_path):
    """Load training information if available"""
    training_info_path = os.path.join(adapter_path, "training_info.json")
    if os.path.exists(training_info_path):
        with open(training_info_path, "r") as f:
            training_info = json.load(f)
        print(f">> Training info loaded: {training_info.get('script_version', 'unknown')} version")
        return training_info
    return {}

def main():
    """Main merge function"""
    # Version info
    log_version_info()
    
    # Configuration
    BASE = "microsoft/DialoGPT-medium"  # Same model used in training
    ADAPTER = "models/out-tinyllama-lora"
    OUT = "models/merged-tinyllama-ft"
    
    print(f">> Base model: {BASE}")
    print(f">> LoRA adapter: {ADAPTER}")
    print(f">> Output model: {OUT}")
    
    # Check if adapter exists
    if not check_adapter_exists(ADAPTER):
        return
    
    # Load training information
    training_info = load_training_info(ADAPTER)
    
    # Create output directory
    os.makedirs(OUT, exist_ok=True)
    print(f">> Output directory created: {OUT}")
    
    # Load model base
    print(">> Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(BASE)
    
    # Load LoRA adapter
    print(">> Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base, ADAPTER)
    
    # Merge and unload
    print(">> Merging LoRA adapter with base model...")
    merged = peft_model.merge_and_unload()
    
    # Save merged model
    print(">> Saving merged model...")
    merged.save_pretrained(OUT)
    
    # Save tokenizer
    print(">> Saving tokenizer...")
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(OUT)
    
    # Save merge information
    merge_info = {
        "script_version": SCRIPT_VERSION,
        "base_model": BASE,
        "adapter_path": ADAPTER,
        "output_path": OUT,
        "merge_time": datetime.now().isoformat(),
        "original_training_info": training_info,
        "device_used": "cpu",  # Merging typically done on CPU
        "torch_version": torch.__version__
    }
    
    with open(os.path.join(OUT, "merge_info.json"), "w") as f:
        json.dump(merge_info, f, indent=2)
    
    print("✅ Modelo mergeado en:", OUT)
    print("✅ Merge info saved to merge_info.json")
    print(f"🎉 Merge completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show file sizes
    try:
        adapter_size = sum(os.path.getsize(os.path.join(ADAPTER, f)) 
                          for f in os.listdir(ADAPTER) 
                          if os.path.isfile(os.path.join(ADAPTER, f)))
        merged_size = sum(os.path.getsize(os.path.join(OUT, f)) 
                         for f in os.listdir(OUT) 
                         if os.path.isfile(os.path.join(OUT, f)))
        
        print(f"\n📊 File sizes:")
        print(f"   LoRA adapter: {adapter_size / 1e6:.1f} MB")
        print(f"   Merged model: {merged_size / 1e6:.1f} MB")
        print(f"   Size ratio: {merged_size / adapter_size:.1f}x larger")
        
    except Exception as e:
        print(f"⚠️  Could not calculate file sizes: {e}")
    
    print(f"""
🚀 Your merged model is ready!
   Location: {OUT}
   Usage: Use this directory as a regular transformers model
   Example: AutoModelForCausalLM.from_pretrained("{OUT}")
""")

if __name__ == "__main__":
    main()
