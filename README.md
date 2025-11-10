# Fine-tuning LoRA en Ubuntu + RTX 4060 Ti

> Proyecto personal documentado como parte de mi portafolio. Implementa un flujo completo de fine-tuning con LoRA/QLoRA sobre modelos modernos (Qwen2.5-7B-Instruct) usando CUDA en Ubuntu, incluyendo preparación del entorno, entrenamiento reproducible y despliegue de inferencia.

## ✨ Resumen del Proyecto
- **Objetivo:** Entrenar un chatbot corporativo capaz de responder procedimientos internos a partir de un dataset de instrucciones propio.
- **Tecnologías:** Python, PyTorch 2.0+, Transformers 4.40+, PEFT 0.10+, TRL, CUDA 12.1+.
- **Hardware:** NVIDIA RTX 4060 Ti (16 GB VRAM) en Ubuntu 22.04+.
- **Características destacadas:**
  - Soporte para QLoRA 4-bit (optimización de memoria)
  - Entrenamiento estable con bfloat16
  - Gradient Checkpointing y optimizaciones de memoria
  - Early Stopping y evaluación por pasos
  - Packing de secuencias opcional
- **Repositorio:** [`andrewvergel/finetuning-linux`](https://github.com/andrewvergel/finetuning-linux)

## 🧱 Arquitectura y Flujo
1. **Bootstrap de entorno** – instalación de drivers, CUDA y dependencias en un equipo limpio.
2. **Dataset JSONL** – prompts internos versionados en `data/instructions.jsonl`.
3. **Script de entrenamiento** – `scripts/finetune_lora.py` (v1.1.1) realiza duplicación inteligente del dataset y ajusta hiperparámetros para escenarios de pocos datos.
4. **Inferencia controlada** – `scripts/inference_lora.py` (v1.0.2) con decodificación determinista para evaluar resultados.
5. **Reportes** – se genera `training_info.json` con metadatos del entrenamiento.

## 🚀 Puesta en Marcha desde Cero
```bash
# 1. Clonar el repositorio
git clone https://github.com/andrewvergel/finetuning-linux.git
cd finetuning-linux

# 2. Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias base
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Instalar dependencias del proyecto
pip install "numpy<2.0" pyarrow==14.0.1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
pip install -r requirements.txt
```
> 💡 Si partes de un servidor recién formateado, instala previamente los drivers NVIDIA, CUDA 12.1 y utilidades del sistema (detallado en secciones posteriores del repositorio original).
> 📦 `requirements.txt` incluye todas las librerías auxiliares; aun así, instalamos `numpy<2.0` y `pyarrow==14.0.1` antes para evitar conflictos conocidos con `datasets` (error `PyExtensionType`).
> 🔍 Antes de entrenar, puedes ejecutar `python scripts/validate_environment.py` para verificar versiones de Python, CUDA, VRAM disponible, dataset y dependencias.
> 🧩 Para modelos Qwen2.5 asegúrate de usar `transformers>=4.40` y `peft>=0.10` (ya fijados en `requirements.txt`).
> 🧾 Cada entrenamiento deja un log detallado en `logs/debug_last_run.log` con métricas y respuestas de validación automática.

## 📚 Dataset de Instrucciones
```bash
mkdir -p data
cat > data/instructions.jsonl << 'JSONL'
{"system":"Eres un asistente experto en procesos internos.","input":"Dame los pasos para conciliar pagos de los lunes.","output":"1) Exporta el CSV del banco.\n2) Ejecuta el job 'reconcile_monday'.\n3) Revisa discrepancias en la tabla 'recon_issues'."}
{"system":"Habla en tono profesional y conciso.","input":"Resume este procedimiento en tres bullets.","output":"• Exportar CSV.\n• Ejecutar job.\n• Validar discrepancias."}
{"system":"Responde siempre con pasos numerados.","input":"¿Cómo abro un ticket de soporte?","output":"1) Entra al helpdesk.\n2) Crea ticket 'Incidente'.\n3) Adjunta evidencias."}
JSONL
```
> ℹ️ `data/instructions.jsonl` **ya viene versionado en este repositorio** y es el único archivo permitido dentro de `data/`. El script de entrenamiento duplica automáticamente el dataset si detecta menos de 200 muestras, pero se recomienda ampliarlo manualmente con más casuísticas para mejorar la diversidad de respuestas.

## 🛠️ Script de Entrenamiento (`scripts/finetune_lora.py` v1.2.0)

### Características Principales
- **Modelo Base:** `Qwen/Qwen2.5-7B-Instruct` por defecto (soporta cualquier modelo compatible con Transformers)
- **Optimizaciones de Memoria:**
  - QLoRA 4-bit activable vía `FT_USE_QLORA=1`
  - Gradient Checkpointing
  - bfloat16 por defecto (óptimo para GPUs Ada/Lovelace)
  - Packing de secuencias opcional (`FT_FORCE_PACKING`)
- **Entrenamiento Estable:**
  - Early Stopping basado en pérdida de validación
  - Evaluación por pasos configurables (`FT_EVAL_STEPS`)
  - Guardado de checkpoints incremental
  - Logs detallados en `logs/debug_last_run.log`
- **Configuración Flexible:**
  - Todas las opciones configurables mediante variables de entorno `FT_*`
  - Soporte para múltiples objetivos LoRA
  - Batch size y acumulación de gradientes configurables

### Flujo de Entrenamiento
1. Carga y validación del dataset desde `data/instructions.jsonl`
2. División automática entrenamiento/validación (85/15% por defecto)
3. Carga del modelo base con optimizaciones de memoria
4. Aplicación de LoRA/QLoRA según configuración
5. Entrenamiento con monitoreo de métricas
6. Evaluación periódica y guardado de checkpoints
7. Generación de informe final con ejemplos de inferencia

### 🧾 Configuración Recomendada (`.env`)

```bash
# Modelo y Datos
FT_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
FT_DATA_PATH=data/instructions.jsonl
FT_OUT_DIR=models/out-qlora
FT_TRUST_REMOTE_CODE=1  # Requerido para Qwen2.5

# Optimización de Memoria
FT_USE_QLORA=1           # Activar QLoRA 4-bit
FT_FORCE_PACKING=0       # Desactivar packing por defecto (más memoria)
FT_GRADIENT_CHECKPOINTING=1

# Hiperparámetros de Entrenamiento
FT_PER_DEVICE_BATCH_SIZE=1
FT_GRADIENT_ACCUMULATION=8
FT_NUM_EPOCHS=5
FT_LEARNING_RATE=2e-5
FT_WARMUP_RATIO=0.1
FT_LR_SCHEDULER=cosine_with_restarts
FT_WEIGHT_DECAY=0.02

# Configuración LoRA
FT_LORA_RANK=8
FT_LORA_ALPHA=16
FT_LORA_DROPOUT=0.05
FT_LORA_TARGET_MODULES=q_proj,v_proj

# Validación y Guardado
FT_EVAL_STEPS=25
FT_SAVE_STEPS=25
FT_SAVE_TOTAL_LIMIT=2
FT_EVAL_MAX_NEW_TOKENS=128
FT_EVAL_SAMPLE_SIZE=3

# Otros
FT_LOGGING_STEPS=10
FT_DATASET_SHUFFLE_SEED=42
FT_VALIDATION_SPLIT=0.15
FT_DEBUG_LOG_FILE=debug_last_run.log
```
> Duplica el archivo como `.env` y personaliza los valores si necesitas cambiar cualquier hiperparámetro sin editar el script.

### 🔍 Interpretación de logs y tuning
#### Pérdida de entrenamiento
- Valores de `loss` entre **4.0–5.0** son típicos para DialoGPT con datasets repetidos. Si cae de ~5.2 a ~4.1 en pocas épocas, la convergencia va bien. Si la pérdida se estanca >3.8 tras 20 épocas, considera subir `LEARNING_RATE` o reducir `LORA_DROPOUT`.
#### Learning rate efectivo
- Con `LEARNING_RATE = 4e-5` deberías ver valores ~3.9e-05 a 4.0e-05 en los logs. Si cae demasiado rápido (<3e-05) en las primeras épocas, sube `WARMUP_RATIO`.
#### Señales de overfitting
- Pérdida de entrenamiento baja, pero validación no mejora o sube → sube `LORA_DROPOUT` o baja `NUM_EPOCHS`.
#### Si el modelo delira
- Repite frases, respuestas circulares o incoherentes → sube `LORA_DROPOUT` a 0.2, baja `NUM_EPOCHS` a 20, y añade más ejemplos únicos al dataset.

#### Fases avanzadas del entrenamiento (épocas 20+)
- Si ves `loss` ~2.0 estable y `learning_rate` ~3e-06, el modelo está cerca del mínimo. 
- Continuar entrenando puede llevar a **overfitting sutil** (responde mejor a ejemplos de entrenamiento pero falla en variaciones). 
- **Criterio de parada:** si `eval_loss` deja de bajar por 3–4 épocas seguidas, detén el entrenamiento. 
- **Si necesitas más calidad:** en lugar de más épocas, amplía el dataset real (no repitas) o prueba un modelo base mayor.

#### ✅ Señales de progreso saludable (épocas 1–5)
- `learning_rate` debería subir de ~6e-06 a ~3e-05 durante las primeras épocas (indica warmup funcionando).
- `eval_loss` debe bajar de forma consistente (ej.: 7.8 → 6.9 entre época 1 y 3).
- `loss` de entrenamiento entre 7.0–8.5 al inicio, bajando gradualmente.

### 🎯 Guía de Ajuste de Hiperparámetros

#### Optimización de Memoria (RTX 4060 Ti 16GB)
- **`FT_USE_QLORA` (1):** Activa cuantización 4-bit (recomendado para modelos >7B)
- **`FT_PER_DEVICE_BATCH_SIZE` (1):** Mantener en 1 para máxima estabilidad
- **`FT_GRADIENT_ACCUMULATION` (8):** Ajustar según VRAM disponible (más alto = mejor uso de GPU)
- **`FT_FORCE_PACKING` (0):** Desactivado por defecto (usa más memoria pero más estable)

#### Rendimiento del Entrenamiento
- **`FT_LORA_RANK` (8):** Dimensión de las matrices de bajo rango
  - *Aumentar* (16-32) para tareas complejas
  - *Reducir* (4-8) si hay problemas de memoria
- **`FT_LEARNING_RATE` (2e-5):** Tasa de aprendizaje base
  - *Aumentar* (3e-5) si la pérdida se estanca
  - *Reducir* (1e-5) si la pérdida es inestable
- **`FT_LORA_ALPHA` (16):** Factor de escalado (normalmente 2× rank)

#### Regularización
- **`FT_LORA_DROPOUT` (0.05):** Regularización para evitar sobreajuste
  - *Aumentar* (0.1-0.2) si el modelo memoriza
  - *Reducir* (0.01) si el aprendizaje es lento
- **`FT_WEIGHT_DECAY` (0.02):** Decaimiento de pesos
  - *Aumentar* (0.05) para más regularización
  - *Reducir* (0.01) si el modelo no converge

#### Evaluación
- **`FT_EVAL_STEPS` (25):** Frecuencia de evaluación
- **`FT_EVAL_SAMPLE_SIZE` (3):** Número de ejemplos para evaluación rápida
- **`FT_EVAL_MAX_NEW_TOKENS` (128):** Longitud máxima de generación en evaluación

## 💬 Script de Inferencia (`scripts/inference_lora.py`)
- Carga el adaptador LoRA desde `models/out-tinyllama-lora`.
- Usa decodificación determinista (sin muestreo) para validar fácilmente regresiones.
- Incluye loop interactivo opcional y estadísticas de uso de GPU.

## ▶️ Ejecución
```bash
# Entrenamiento
python scripts/finetune_lora.py

# Inferencia inicial
python scripts/inference_lora.py
```
> El entrenamiento guarda pesos LoRA en `models/out-tinyllama-lora`. Puedes fusionarlos con el modelo base usando `scripts/merge_adapter.py` si necesitas un único checkpoint.

## 🧩 Estructura del Proyecto
```
finetuning-linux/
├── data/
│   └── instructions.jsonl           # Dataset versionado
├── logs/
│   └── debug_last_run.log          # Log detallado del último entrenamiento
├── models/                          # Salidas de entrenamiento (ignorado en git)
│   └── out-qlora/                  # Checkpoints del modelo
│       ├── adapter_model.bin       # Pesos del adaptador LoRA
│       ├── config.json             # Configuración del modelo
│       └── training_info.json      # Métricas y metadatos
├── scripts/
│   ├── finetune_lora.py            # Entrenamiento LoRA/QLoRA (v1.2.0)
│   ├── inference_lora.py           # Inferencia con adaptadores
│   ├── merge_adapter.py            # Fusión de adaptadores con el modelo base
│   └── validate_environment.py     # Verificación del entorno
├── .env.example                    # Plantilla de configuración
├── .gitignore
└── README.md                       # Este documento
```