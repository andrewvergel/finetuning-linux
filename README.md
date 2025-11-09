# Fine-tuning LoRA en Ubuntu + RTX 4060 Ti

> Proyecto personal documentado como parte de mi portafolio. Implementa un flujo completo de fine-tuning con LoRA sobre DialoGPT usando CUDA en Ubuntu, incluyendo preparación del entorno, entrenamiento reproducible y despliegue de inferencia.

## ✨ Resumen del Proyecto
- **Objetivo:** entrenar un chatbot corporativo capaz de responder procedimientos internos a partir de un dataset de instrucciones propio.
- **Tecnologías:** Python, PyTorch 2.8, Transformers 4.35, TRL 0.7, LoRA (PEFT), CUDA 12.1.
- **Hardware:** NVIDIA RTX 4060 Ti (16 GB VRAM) en Ubuntu 22.04.
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

# 3. Actualizar pip e instalar dependencias
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0" pyarrow==14.0.1
pip install -r requirements.txt
```
> 💡 Si partes de un servidor recién formateado, instala previamente los drivers NVIDIA, CUDA 12.1 y utilidades del sistema (detallado en secciones posteriores del repositorio original).
> 📦 `requirements.txt` incluye todas las librerías auxiliares; aun así, instalamos `numpy<2.0` y `pyarrow==14.0.1` antes para evitar conflictos conocidos con `datasets` (error `PyExtensionType`).
> 🔍 Antes de entrenar, puedes ejecutar `python scripts/validate_environment.py` para verificar versiones de Python, CUDA, VRAM disponible, dataset y dependencias.
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

## 🛠️ Script de Entrenamiento (`scripts/finetune_lora.py`)
- Basado en LoRA (r=32) sobre las capas `c_attn` y `c_proj` de DialoGPT-medium (ajustable por constantes).
- El entrenamiento usa por defecto `data/instructions.jsonl` (puedes sobreescribirlo con la variable `FINETUNE_DATA_PATH`).
- Base por defecto: `Qwen/Qwen2.5-7B-Instruct` (activando `FT_TRUST_REMOTE_CODE=1` en `.env`).
- Entrenamiento altamente regularizado: batch efectivo 32 (4×8), 8 épocas, scheduler `cosine` (warmup 15%) y weight decay 0.01.
- Genera `training_info.json` con metadatos y deja un log detallado en `logs/debug_last_run.log`.
- Reserva automáticamente 15% para validación, corre evaluación al final de cada época y guarda el mejor checkpoint según `eval_loss`.
- Ejecuta una evaluación rápida al final tomando 12 ejemplos del split de validación (o un fallback predefinido) y deja la comparación esperada/obtenida en el log.
- Soporta variables de entorno (`FT_*`). Puedes crear un `.env` en la raíz con los valores que necesites.

### 🧾 Ejemplo de `.env`
```bash
FT_MODEL_ID=microsoft/DialoGPT-medium
FT_DATA_PATH=data/instructions.jsonl
FT_OUT_DIR=models/out-tinyllama-lora
FT_DATASET_MIN_EXAMPLES=240
FT_PER_DEVICE_BATCH_SIZE=1
FT_GRADIENT_ACCUMULATION=12
FT_NUM_EPOCHS=12
FT_LEARNING_RATE=2e-5
FT_WARMUP_RATIO=0.1
FT_LR_SCHEDULER=linear
FT_WEIGHT_DECAY=0.1
FT_LORA_RANK=32
FT_LORA_ALPHA=32
FT_LORA_DROPOUT=0.3
FT_LORA_TARGET_MODULES=c_attn,c_proj
FT_LOGGING_STEPS=5
FT_SAVE_STRATEGY=epoch
FT_SAVE_TOTAL_LIMIT=3
FT_DATASET_SHUFFLE_SEED=42
FT_VALIDATION_SPLIT=0.2
FT_DEBUG_LOG_FILE=debug_last_run.log
FT_EVAL_MAX_NEW_TOKENS=220
FT_EVAL_SAMPLE_SIZE=10
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

### 📊 Guía rápida de hiperparámetros
- `DATASET_MIN_EXAMPLES = 240` → número mínimo de muestras tras repetir el split de entrenamiento. *Subirlo* (300) añade más iteraciones; *bajarlo* (180) cuando agregues más ejemplos únicos.
- `PER_DEVICE_BATCH_SIZE = 4` → muestras procesadas por GPU antes de acumular gradientes. Con QLoRA 4-bit el consumo de VRAM se mantiene estable. *Subirlo* (6) si dispones de más VRAM; *bajarlo* (2) para margen extra.
- `GRADIENT_ACCUMULATION = 8` → batch efectivo 32 (4×8). *Subirlo* (10) suaviza más los gradientes; *bajarlo* (6) acelera cuando agregues más datos.
- `NUM_EPOCHS = 8` → con repetición 12× cada muestra se ve unas 2 880 veces. *Subirlo* (10) si el `eval_loss` mejora; *bajarlo* (6) cuando amplíes el dataset.
- `LEARNING_RATE = 1e-4` → LR recomendado por Qwen para LoRA. *Subirlo* (1.2e-4) si el loss se estanca; *bajarlo* (8e-5) si la validación oscila.
- `WARMUP_RATIO = 0.15` → arranque suave (~15% de los pasos). *Subirlo* (0.2) si el loss inicial explota; *bajarlo* (0.1) cuando uses LR menores.
- `LORA_DROPOUT = 0.15` → regularización sobre capas adaptadas. *Subirlo* (0.2) si aún repite; *bajarlo* (0.1) cuando tengas más ejemplos únicos.
- `EVAL_SAMPLE_SIZE = 10` → cantidad de ejemplos del split de validación usados en la evaluación rápida.

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
├── models/                          # Salidas de entrenamiento (ignorado en git)
├── scripts/
│   ├── finetune_lora.py             # Entrenamiento LoRA (v1.1.1)
│   ├── inference_lora.py            # Inferencia determinista (v1.0.2)
│   └── validate_environment.py      # Checklist opcional de diagnóstico
├── .gitignore
└── README.md (este documento)
```