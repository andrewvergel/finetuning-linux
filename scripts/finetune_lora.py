import os, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# Usar un modelo compatible con transformers 4.25.1
MODEL_ID = "gpt2"  # Modelo alternativo compatible
DATA_PATH = "data/instructions.jsonl"
OUT_DIR = "out-tinyllama-lora"

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(">> Device:", device)

# Carga dataset
ds = load_dataset("json", data_files=DATA_PATH)["train"]
print(f">> Dataset loaded: {len(ds)} examples")

# Tokenizer + modelo base (usar GPT-2 que es compatible)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f">> Tokenizer loaded, pad_token: {tok.pad_token}")

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.to(device)
print(f">> Model loaded and moved to {device}")

# Config LoRA
peft_cfg = LoraConfig(
    r=32, lora_alpha=32, lora_dropout=0.05,
    target_modules=["c_attn"],  # target modules para GPT-2
    bias="none", task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_cfg)

# Preparar datos - formato compatible con Trainer
def format_example(ex):
    sys = ex.get("system","Eres un asistente útil y conciso.")
    prompt = f"System: {sys}\nUser: {ex['input']}\nAssistant: {ex['output']}"
    # Tokenizar con parámetros fijos para asegurar longitudes consistentes
    encoded = tok(
        prompt, 
        return_tensors="pt", 
        padding="max_length",     # PAD a max_length
        truncation=True,          # TRUNCATE a max_length
        max_length=256           # Longitud fija
    )
    input_ids = encoded["input_ids"][0]  # Remover batch dimension
    attention_mask = encoded["attention_mask"][0]  # Remover batch dimension
    
    # Para causal LM, los labels son los mismos que input_ids
    labels = input_ids.clone()
    
    return {
        "input_ids": input_ids.tolist(),  # Convertir a lista para compatibilidad con datasets
        "attention_mask": attention_mask.tolist(),
        "labels": labels.tolist()
    }

print(">> Processing dataset...")
processed_ds = ds.map(format_example, remove_columns=ds.column_names)
print(f">> Dataset after processing: {len(processed_ds)} examples")
if len(processed_ds) > 0:
    print(f">> First example keys: {list(processed_ds[0].keys())}")
    print(f">> Input IDs length: {len(processed_ds[0]['input_ids'])}")
    print(f">> Labels length: {len(processed_ds[0]['labels'])}")

# No usar data_collator personalizado, dejar que Trainer maneje los datos ya procesados
# Simplificar para usar el approach estándar de transformers

# Configurar Trainer para manejar la tokenización y etiquetas
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=4,          # Batch más grande para velocidad
    gradient_accumulation_steps=2,          # Menos acumulación
    learning_rate=1e-3,                     # Learning rate más alto
    num_train_epochs=2,                     # Menos épocas para velocidad
    logging_steps=50,                       # Logging menos frecuente
    save_steps=100,                         # Guardar menos frecuentemente
    evaluation_strategy="no",
    warmup_ratio=0.03,                      # Warmup rápido
    dataloader_pin_memory=False,
    remove_unused_columns=False,  # Importante para mantener las columnas
    logging_dir="./logs",               # Logs para monitorear
    # Parámetros adicionales para mejor convergencia
    weight_decay=0.01,                      # Regularización
    adam_epsilon=1e-8,                     # Épsilon para Adam
    max_grad_norm=1.0,                      # Clipping de gradientes
)

trainer = Trainer(
    model=model,
    tokenizer=tok,
    train_dataset=processed_ds,
    args=training_args,
)

trainer.train()
trainer.model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print("✅ Adaptador LoRA guardado en:", OUT_DIR)
