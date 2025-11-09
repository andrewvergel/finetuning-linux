¡Perfecto! Esta guía te permite hacer fine-tuning nativo en Ubuntu con GPU NVIDIA usando CUDA para aceleración completa.

⸻

# Sistema de Fine-tuning LoRA para Ubuntu + GPU NVIDIA

## 🚀 INSTALACIÓN COMPLETA DESDE CERO

### Paso 1) Actualización del Sistema

```bash
# ===== ACTUALIZAR SISTEMA COMPLETO =====
sudo apt update && sudo apt upgrade -y

# ===== INSTALAR DEPENDENCIAS DEL SISTEMA =====
sudo apt install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    htop \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    libjpeg-dev \
    zlib1g-dev \
    nvidia-driver-535  # Driver NVIDIA
```

**⚠️ IMPORTANTE**: Reinicia el sistema después de instalar el driver:
```bash
sudo reboot
```

### Paso 2) Verificación de GPU NVIDIA

```bash
# Verificar que la GPU está detectada
lspci | grep -i nvidia

# Verificar driver instalado
nvidia-smi
# Deberías ver información de tu GPU NVIDIA

# Verificar versión del driver
nvidia-smi | grep "Driver Version"
```

### Paso 3) Instalación de CUDA Toolkit

#### Opción A: Instalación vía APT (Recomendado)
```bash
# Agregar repositorio NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Instalar CUDA 12.1
sudo apt-get -y install cuda-toolkit-12-1
```

#### Opción B: Instalación manual
```bash
# Descargar e instalar CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```

### Paso 4) Configuración de Variables de Entorno

```bash
# Agregar CUDA al PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verificar instalación CUDA
nvcc --version
# Deberías ver: "Cuda compilation tools, release 12.1"
```

### Paso 5) Configuración del Entorno Python

```bash
# Verificar Python
python3 --version
pip3 --version

# Crear virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Actualizar pip
pip install --upgrade pip
```

### Paso 6) Instalación de PyTorch con CUDA

```bash
# Instalar PyTorch con soporte CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verificar que CUDA funciona
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU CUDA: {torch.cuda.get_device_name(0)}')
    print(f'Versión CUDA: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    # Test de rendimiento
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    import time
    start = time.time()
    z = torch.mm(x, y)
    torch.cuda.synchronize()
    end = time.time()
    print(f'Matrix multiplication (1000x1000): {(end-start)*1000:.2f}ms')
else:
    print('❌ CUDA no está disponible')
"
```

### Paso 7) Instalación de Librerías de Machine Learning

```bash
# Stack de Hugging Face (versiones compatibles)
pip install transformers==4.25.1
pip install datasets==2.13.2
pip install peft==0.3.0
pip install accelerate==0.20.3
pip install trl==0.7.4
pip install sentencepiece
pip install safetensors

# Librerías adicionales útiles
pip install jupyterlab
pip install matplotlib seaborn pandas numpy
pip install tqdm tensorboard
```

### Paso 8) Verificación Final del Entorno

```bash
# Test completo de todas las librerías
python -c "
import torch
import transformers
import datasets
from peft import PeftModel
print('✅ Todas las librerías importadas correctamente')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Datasets: {datasets.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"
```

⸻

## 📁 CONFIGURACIÓN DEL PROYECTO

### Estructura de Directorios

```bash
# Crear estructura del proyecto
mkdir -p ft-linux-gpu/{data,scripts,models,logs}
cd ft-linux-gpu

# Activar virtualenv
source ../.venv/bin/activate
```

### Dataset de Ejemplo

```bash
# Crear dataset de instrucciones
cat > data/instructions.jsonl << 'JSONL'
{"system":"Eres un asistente experto en procesos internos.","input":"Dame los pasos para conciliar pagos de los lunes.","output":"1) Exporta el CSV del banco.\n2) Ejecuta el job 'reconcile_monday'.\n3) Revisa discrepancias en la tabla 'recon_issues'."}
{"system":"Habla en tono profesional y conciso.","input":"Resume este procedimiento en tres bullets.","output":"• Exportar CSV.\n• Ejecutar job.\n• Validar discrepancias."}
{"system":"Responde siempre con pasos numerados.","input":"¿Cómo abro un ticket de soporte?","output":"1) Entra al helpdesk.\n2) Crea ticket 'Incidente'.\n3) Adjunta evidencias."}
JSONL
```

⸻

## 🔧 SCRIPTS DE FINE-TUNING

### Script de Fine-tuning (Optimizado para CUDA)

```python
# scripts/finetune_lora.py
import os, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# Verificar CUDA disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">> Device detectado: {device}")
print(f">> GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

MODEL_ID = "gpt2"  # Modelo base ligero
DATA_PATH = "data/instructions.jsonl"
OUT_DIR = "models/out-tinyllama-lora"

# Cargar dataset
ds = load_dataset("json", data_files=DATA_PATH)["train"]

# Tokenizer + modelo base
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.to(device)

# Configuración LoRA optimizada para GPU NVIDIA
peft_cfg = LoraConfig(
    r=32, lora_alpha=32, lora_dropout=0.05,
    target_modules=["c_attn"],  # target modules para GPT-2
    bias="none", task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_cfg)

# Formateo de ejemplos
def format_example(ex):
    sys = ex.get("system","Eres un asistente útil y conciso.")
    prompt = f"System: {sys}\nUser: {ex['input']}\nAssistant: {ex['output']}"
    return {"text": prompt}

ds = ds.map(format_example, remove_columns=ds.column_names)

# Configuración optimizada para GPU CUDA
sft_args = SFTConfig(
    output_dir=OUT_DIR,
    per_device_train_batch_size=4,  # Más alto para GPU
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    eval_strategy="no",
    max_seq_length=1024,  # Más largo para GPU
    packing=True,
    warmup_ratio=0.03,
    dataloader_pin_memory=True,  # Optimización para GPU
    report_to="tensorboard",
    logging_dir="logs",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds,
    args=sft_args,
    formatting_func=lambda ex: ex["text"],
)

# Entrenar modelo
trainer.train()
trainer.model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print("✅ Adaptador LoRA guardado en:", OUT_DIR)
print("✅ Logs disponibles en:", "logs/")
```

### Script de Inferencia

```python
# scripts/inference_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Auto-detectar device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">> Device detectado: {device}")

BASE = "gpt2"
ADAPTER = "models/out-tinyllama-lora"

# Cargar tokenizer
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Cargar modelo base + LoRA
base = AutoModelForCausalLM.from_pretrained(BASE)
base = base.to(device)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

def chat(user, system="Eres un asistente profesional y conciso."):
    prompt = f"System: {system}\nUser: {user}\nAssistant:"
    ids = tok(prompt, return_tensors="pt").to(device)
    
    gen = model.generate(
        **ids, 
        max_new_tokens=300,
        do_sample=True, 
        top_p=0.9, 
        temperature=0.7,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id
    )
    
    response = tok.decode(gen[0], skip_special_tokens=True)
    # Extraer solo la respuesta del assistant
    if "Assistant:" in response:
        response = response.split("Assistant:")[1].strip()
    
    print("🤖 Respuesta:", response)
    return response

if __name__ == "__main__":
    print("=== CHAT CON MODELO FINE-TUNED ===")
    test_query = "Dame un checklist de conciliación de pagos de los lunes."
    print(f"Usuario: {test_query}")
    chat(test_query)
```

### Script de Merge de LoRA

```python
# scripts/merge_adapter.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE = "gpt2"
ADAPTER = "models/out-tinyllama-lora"
OUT = "models/merged-tinyllama-ft"

print("Cargando modelo base...")
base = AutoModelForCausalLM.from_pretrained(BASE)
peft_model = PeftModel.from_pretrained(base, ADAPTER)
merged = peft_model.merge_and_unload()

os.makedirs(OUT, exist_ok=True)
merged.save_pretrained(OUT)
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.save_pretrained(OUT)
print("✅ Modelo mergeado en:", OUT)
```

⸻

## 🚀 EJECUCIÓN DEL FINE-TUNING

### Entrenar el Modelo

```bash
# Activar entorno
source .venv/bin/activate

# Ejecutar fine-tuning
python scripts/finetune_lora.py

# Deberías ver:
# >> Device detectado: cuda
# >> GPU: NVIDIA GeForce RTX XXXX
# >> Training completed successfully
```

### Probar Inferencia

```bash
# Probar el modelo entrenado
python scripts/inference_lora.py

# Deberías ver una respuesta específica a tu dominio
# 🤖 Respuesta: 1) Exporta el CSV del banco. 2) Ejecuta el job 'reconcile_monday'...
```

### (Opcional) Merge del Modelo

```bash
# Crear modelo único sin LoRA
python scripts/merge_adapter.py

# Resultado en ./models/merged-tinyllama-ft
```

⸻

## 📊 MONITOREO Y RENDIMIENTO

### Monitoreo en Tiempo Real

```bash
# Verificar uso de GPU durante entrenamiento
nvidia-smi -l 1

# Monitorear temperatura
watch -n 1 nvidia-smi

# Verificar logs de entrenamiento
tail -f logs/events.out.tfevents.*
```

### Optimizaciones para GPU

```python
# Configuraciones recomendadas por VRAM:

# GPU con 6GB VRAM
per_device_train_batch_size=2
gradient_accumulation_steps=8
max_seq_length=512

# GPU con 8GB VRAM  
per_device_train_batch_size=4
gradient_accumulation_steps=4
max_seq_length=1024

# GPU con 12GB+ VRAM
per_device_train_batch_size=8
gradient_accumulation_steps=2
max_seq_length=2048
```

### Test de Rendimiento

```bash
# Test de velocidad de entrenamiento
python -c "
import torch, time
if torch.cuda.is_available():
    print('Testing GPU performance...')
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    
    # Warm-up
    for _ in range(10):
        z = torch.mm(x, y)
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        z = torch.mm(x, y)
        torch.cuda.synchronize()
    end = time.time()
    
    print(f'Average time: {(end-start)/100*1000:.2f}ms per multiplication')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA no disponible')
"
```

⸻

## 🛠️ SOLUCIÓN DE PROBLEMAS

### Problemas Comunes

**❌ "NVIDIA-SMI not found"**
```bash
# Reinstalar driver
sudo apt install nvidia-driver-535
sudo reboot
```

**❌ "CUDA not found"**
```bash
# Verificar instalación
ls /usr/local/cuda*
# Reinstalar si es necesario
sudo apt install cuda-toolkit-12-1
```

**❌ "Permission denied" en GPU**
```bash
# Verificar permisos
ls -l /dev/nvidia*
# Agregar usuario a grupos
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER
# Reiniciar sesión
```

**❌ "Out of Memory" durante training**
```bash
# Verificar memoria GPU
nvidia-smi -l 1
# Reducir batch size en el script
```

**❌ "Module not found" errores**
```bash
# Recrear virtualenv
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.25.1 datasets==2.13.2 peft==0.3.0 accelerate==0.20.3 trl==0.7.4 sentencepiece safetensors
```

### Comandos de Diagnóstico

```bash
#!/bin/bash
# diagnostics.sh - Script de diagnóstico completo

echo "=== INFORMACIÓN DEL SISTEMA ==="
lsb_release -a
uname -r

echo -e "\n=== GPU NVIDIA ==="
nvidia-smi

echo -e "\n=== CUDA ==="
nvcc --version
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda

echo -e "\n=== PYTHON ENVIRONMENT ==="
which python
python --version
which pip
pip --version

echo -e "\n=== TEST DE PYTORCH CON CUDA ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"
```

⸻

## 🎯 PRÓXIMOS PASOS

### Para Adaptar a tu Dominio

1. **Preparar tu Dataset**: Convierte tu documentación/procedimientos a formato JSONL:
   ```json
   {"system": "Tus instrucciones de sistema", "input": "Pregunta frecuente", "output": "Respuesta esperada"}
   ```

2. **Entrenar con tus Datos**: Reemplaza `data/instructions.jsonl` y ejecuta el fine-tuning

3. **Evaluar Resultados**: Prueba el modelo con casos reales de tu dominio

4. **Optimizar**: Ajusta hiperparámetros según resultados

### Para Producción

- **Quantización**: Usa `bitsandbytes` para modelos más pequeños
- **Servidor de Inferencia**: Implementa con FastAPI + CUDA
- **Monitoreo**: Logs de TensorBoard para seguimiento
- **Backup**: Guarda checkpoints intermedios

¡Listo! Tienes un sistema completo de fine-tuning optimizado para Ubuntu + GPU NVIDIA. 🚀
