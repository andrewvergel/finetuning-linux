from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER = "out-tinyllama-lora"
OUT = "merged-tinyllama-ft"

base = AutoModelForCausalLM.from_pretrained(BASE)
peft_model = PeftModel.from_pretrained(base, ADAPTER)
merged = peft_model.merge_and_unload()

os.makedirs(OUT, exist_ok=True)
merged.save_pretrained(OUT)
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.save_pretrained(OUT)
print("✅ Modelo mergeado en:", OUT)
