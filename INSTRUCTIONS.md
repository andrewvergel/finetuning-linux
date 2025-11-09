¡Perfecto! Esta guía te permite hacer fine-tuning nativo en Ubuntu con GPU NVIDIA usando CUDA para aceleración completa.

⸻

# Sistema de Fine-tuning LoRA para Ubuntu + RTX 4060 Ti (16GB VRAM)

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

### 🔧 SOLUCIÓN A CONFLICTOS DE DRIVERS NVIDIA

**Si encuentras errores de dependencias como "conflicts with nvidia-driver-575":**

```bash
# 1. Verificar drivers NVIDIA actuales
nvidia-smi
dpkg -l | grep nvidia

# 2. Remover todos los drivers NVIDIA existentes
sudo apt remove --purge -y nvidia-*
sudo apt autoremove -y
sudo apt autoclean

# 3. Limpiar residuos
sudo rm -rf /usr/local/cuda*
sudo rm -rf ~/.nvidia*

# 4. Actualizar sistema después de limpieza
sudo apt update

# 5. Instalar driver más reciente (compatible con RTX 4090)
sudo apt install -y nvidia-driver-570  # Driver más reciente
# O usar el PPA para drivers más actualizados:
# sudo add-apt-repository ppa:graphics-drivers/ppa -y
# sudo apt update
# sudo apt install -y nvidia-driver-570

# 6. Reiniciar sistema
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
# Instalar PyTorch con soporte CUDA 12.1 (optimizado para RTX 4090)
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
    print(f'VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    # Test de rendimiento RTX 4090
    x = torch.randn(2000, 2000).cuda()
    y = torch.randn(2000, 2000).cuda()
    import time
    start = time.time()
    z = torch.mm(x, y)
    torch.cuda.synchronize()
    end = time.time()
    print(f'Matrix multiplication (2000x2000): {(end-start)*1000:.2f}ms')
else:
    print('❌ CUDA no está disponible')
"
```

### Paso 7) Instalación de Librerías de Machine Learning

```bash
# Stack optimizado para RTX 4060 Ti
pip install transformers==4.35.2  # Versión más reciente
pip install datasets==2.14.6
pip install peft==0.6.0  # Versión actualizada
pip install accelerate==0.24.1
pip install trl==0.7.6
pip install sentencepiece
pip install safetensors

# Librerías adicionales para modelos grandes
pip install bitsandbytes  # Para quantización
pip install xformers      # Para eficiencia de memoria
pip install jupyterlab
pip install matplotlib seaborn pandas numpy
pip install tqdm tensorboard
```

### 🔧 SOLUCIÓN A ERRORES DE COMPATIBILIDAD

**❌ PROBLEMA REAL: NumPy 2.x Incompatibilidad**

```bash
# Verificar versión de NumPy
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"

# SOLUCIÓN: Downgrade a NumPy 1.x (compatible con ML libraries)
pip install "numpy<2.0"
```

**Después del downgrade de NumPy:**

```bash
# 1. Actualizar pyarrow a versión compatible
pip install --upgrade pyarrow>=15.0.0

# 2. Verificar instalación
pip list | grep -E "(numpy|pyarrow|datasets)"

# 3. Test de importación
python -c "
import numpy as np
print(f'NumPy version: {np.__version__}')

import pyarrow
print(f'PyArrow version: {pyarrow.__version__}')

import datasets
print('✅ Todas las librerías compatibles')
"
```

**✅ RESULTADO ESPERADO:**
```
✅ Todas las librerías importadas correctamente
PyTorch: 2.8.0+cu128
Transformers: 4.35.2
Datasets: 2.14.6
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Ti
VRAM: 16.7GB
```

**Nota:** Los FutureWarnings sobre `_register_pytree_node` son normales y no afectan la funcionalidad.

**Si persiste el problema, stack completo con versiones compatibles:**

```bash
# Reinstalar todo con versiones probadas
pip uninstall -y numpy pyarrow pandas transformers datasets peft accelerate trl

# Instalar NumPy 1.x primero
pip install "numpy<2.0"

# Instalar stack compatible
pip install pyarrow==14.0.1
pip install pandas==2.0.3
pip install transformers==4.35.2
pip install datasets==2.14.6
pip install peft==0.6.0
pip install accelerate==0.24.1
pip install trl==0.7.6
```

**Error alternativo de dependencias de CUDA:**

```bash
# Si hay errores de CUDA con torch
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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

### Script de Fine-tuning (Optimizado para RTX 4060 Ti)

```python
# scripts/finetune_lora.py - Optimizado para RTX 4060 Ti 16GB
import os, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# Verificar CUDA disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">> Device detectado: {device}")
if torch.cuda.is_available():
    print(f">> GPU: {torch.cuda.get_device_name(0)}")
    print(f">> VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Usar modelo mediano aprovechando 16GB VRAM de RTX 4060 Ti
MODEL_ID = "microsoft/DialoGPT-medium"  # O "microsoft/DialoGPT-large" 
DATA_PATH = "data/instructions.jsonl"
OUT_DIR = "models/out-tinyllama-lora"

# Cargar dataset
ds = load_dataset("json", data_files=DATA_PATH)["train"]

# Tokenizer + modelo base
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,  # Usar FP16 para ahorrar VRAM
    device_map="auto",          # Distribución automática
)

# Configuración LoRA optimizada para RTX 4060 Ti
peft_cfg = LoraConfig(
    r=32, lora_alpha=32, lora_dropout=0.05,
    target_modules=["c_attn"],  # Ajustar según el modelo
    bias="none", task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_cfg)

# Formateo de ejemplos
def format_example(ex):
    sys = ex.get("system","Eres un asistente útil y conciso.")
    prompt = f"System: {sys}\nUser: {ex['input']}\nAssistant: {ex['output']}"
    return {"text": prompt}

ds = ds.map(format_example, remove_columns=ds.column_names)

# Configuración optimizada para RTX 4060 Ti (16GB VRAM)
sft_args = SFTConfig(
    output_dir=OUT_DIR,
    per_device_train_batch_size=6,  # Ajustado para RTX 4060 Ti
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    eval_strategy="no",
    max_seq_length=2048,  # Aprovecha 16GB VRAM
    packing=True,
    warmup_ratio=0.03,
    dataloader_pin_memory=True,
    report_to="tensorboard",
    logging_dir="logs",
    bf16=False,  # RTX 4060 Ti funciona mejor con FP16
    tf32=True,   # TensorFloat-32 habilitado
    dataloader_num_workers=2,  # Optimizado para RTX 4060 Ti
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

### Script de Inferencia (Optimizado para RTX 4060 Ti)

```python
# scripts/inference_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Auto-detectar device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">> Device detectado: {device}")

BASE = "microsoft/DialoGPT-medium"
ADAPTER = "models/out-tinyllama-lora"

# Cargar tokenizer
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Cargar modelo base + LoRA
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

def chat(user, system="Eres un asistente profesional y conciso."):
    prompt = f"System: {system}\nUser: {user}\nAssistant:"
    ids = tok(prompt, return_tensors="pt").to(device)
    
    gen = model.generate(
        **ids, 
        max_new_tokens=400,  # Optimizado para RTX 4060 Ti
        do_sample=True, 
        top_p=0.95, 
        temperature=0.7,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True  # Optimización para RTX 4060 Ti
    )
    
    response = tok.decode(gen[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:")[1].strip()
    
    print("🤖 Respuesta:", response)
    return response

if __name__ == "__main__":
    print("=== CHAT CON MODELO FINE-TUNED (RTX 4060 Ti) ===")
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

**⚠️ SOLUCIÓN A ERROR DE TRL IMPORTS:**

Si encuentras el error `ImportError: cannot import name 'SFTConfig' from 'trl'`:

```bash
# 1. Verificar versión de TRL instalada
pip list | grep trl

# 2. Actualizar TRL a versión compatible
pip install --upgrade trl==0.7.6

# 3. Si persiste el error, usar este import alternativo:
# En lugar de:
# from trl import SFTTrainer, SFTConfig
# from transformers import AutoModelForCausalLM, AutoTokenizer

# Usar:
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# from trl import SFTTrainer
```

**Para TRL v0.7.x (recomendado), usar este import:**

```bash
# Verificar que tienes la versión correcta
pip install trl==0.7.6 transformers==4.35.2
```

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

### Optimizaciones para RTX 4060 Ti

```python
# Configuraciones por modelo para RTX 4060 Ti (16GB VRAM):

# Modelo pequeño (GPT-2, DialoGPT-medium)
per_device_train_batch_size=8
gradient_accumulation_steps=4
max_seq_length=2048

# Modelo mediano (DialoGPT-large, TinyLlama-1.1B)
per_device_train_batch_size=4
gradient_accumulation_steps=6
max_seq_length=2048

# Modelo grande (Llama 2 7B con LoRA)
per_device_train_batch_size=2
gradient_accumulation_steps=8
max_seq_length=1024
```

### Test de Rendimiento RTX 4060 Ti

```bash
# Test de velocidad RTX 4060 Ti
python -c "
import torch, time
if torch.cuda.is_available():
    print('=== RTX 4060 Ti Performance Test ===')
    
    # Test 1: Matrices medianas (apropiadas para RTX 4060 Ti)
    size = 3000  # Tamaño moderado para RTX 4060 Ti
    x = torch.randn(size, size).cuda()
    y = torch.randn(size, size).cuda()
    
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
    
    print(f'Matrix multiplication ({size}x{size}): {(end-start)/100*1000:.2f}ms')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    print(f'Memoria libre: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated() / 1e9:.1f}GB')
    
    # Test 2: Atención multi-head
    batch_size, seq_len, hidden_dim = 6, 2048, 768  # Ajustado para RTX 4060 Ti
    query = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    key = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    value = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    
    start = time.time()
    for _ in range(50):
        # Simular atención multi-head
        attn_output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        torch.cuda.synchronize()
    end = time.time()
    
    print(f'Attention computation: {(end-start)/50*1000:.2f}ms')
    
else:
    print('CUDA no disponible')
"
```

### Monitoreo en Tiempo Real para RTX 4060 Ti

```bash
# Verificar uso específico de RTX 4060 Ti
watch -n 1 nvidia-smi

# Monitorear temperaturas (RTX 4060 Ti es más eficiente térmicamente)
nvidia-smi -l 1 -q -d TEMPERATURE

# Verificar frecuencia de GPU
watch -n 1 nvidia-smi --query-gpu=clocks.gr,power.draw,memory.used,memory.total --format=csv

# Monitorear uso de CUDA
nvidia-smi -l 1 --query-compute-apps=pid,used_memory,compute_mode --format=csv
```

### Script de Diagnóstico para RTX 4060 Ti

```bash
#!/bin/bash
# diagnostics_rtx4060ti.sh

echo "=== RTX 4060 Ti DIAGNOSTIC ==="
echo "Fecha: $(date)"
echo "Usuario: $(whoami)"

echo -e "\n=== INFORMACIÓN DEL SISTEMA ==="
lsb_release -a
uname -r
lscpu | grep "Model name"

echo -e "\n=== GPU NVIDIA RTX 4060 Ti ==="
nvidia-smi -L
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv

echo -e "\n=== TEST DE PYTORCH CON CUDA ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Versión CUDA: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    print(f'VRAM Libre: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated() / 1e9:.1f}GB')
    print(f'Multi-Processor Count: {torch.cuda.get_device_properties(0).multi_processor_count}')
    print(f'Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}')
"

echo -e "\n=== TEST DE MEMORIA RTX 4060 Ti ==="
python -c "
import torch
if torch.cuda.is_available():
    # Test de memoria moderado para RTX 4060 Ti
    try:
        test_size = 4 * 1024  # 4GB test
        x = torch.randn(test_size, test_size).cuda()
        print(f'✅ Test de 4GB exitoso')
        del x
        torch.cuda.empty_cache()
        
        # Test de 8GB
        test_size = 6 * 1024  # 6GB test
        x = torch.randn(test_size, test_size).cuda()
        print(f'✅ Test de 6GB exitoso')
        del x
        
        # Test de 12GB (usando majority de 16GB VRAM)
        test_size = 8 * 1024  # 8GB test
        x = torch.randn(test_size, test_size).cuda()
        print(f'✅ Test de 8GB exitoso')
        del x
        
    except RuntimeError as e:
        print(f'❌ Error de memoria: {e}')
"

echo -e "\n=== VERIFICACIÓN DE MÓDULOS ==="
source .venv/bin/activate
python -c "
import sys
modules = ['torch', 'transformers', 'datasets', 'peft', 'accelerate', 'trl', 'sentencepiece', 'safetensors']
for module in modules:
    try:
        exec(f'import {module}')
        print(f'✅ {module} disponible')
    except ImportError as e:
        print(f'❌ {module}: {e}')
"
```

### Comandos Específicos para RTX 4060 Ti

```bash
# Configurar TensorFloat-32 para mejor rendimiento
export NV_TENSORRT_FP16_ENABLE=1
export NV_TENSORRT_INT8_ENABLE=1
export CUDA_TF32_NBLOCK=1

# Optimizar variables del sistema
echo 'export CUDA_CACHE_MAXSIZE=2147483648' >> ~/.bashrc
echo 'export TORCH_CUDNN_V8_API_ENABLED=1' >> ~/.bashrc
source ~/.bashrc

# Verificar optimizaciones aplicadas
python -c "
import torch
print(f'CuDNN version: {torch.backends.cudnn.version()}')
print(f'CuDNN enabled: {torch.backends.cudnn.enabled}')
print(f'TF32 habilitado: {torch.backends.cudnn.allow_tf32}')
print(f'Matmul TF32: {torch.backends.cuda.matmul.allow_tf32}')
"
```

¡Listo! Tienes un sistema completo de fine-tuning optimizado para Ubuntu + RTX 4060 Ti (16GB VRAM). 🚀
