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
- Duplica datasets pequeños hasta ~240 ejemplos para dar más iteraciones útiles.
- Entrenamiento balanceado: batch efectivo 12 (4×3), 30 épocas, scheduler `constant_with_warmup` (warmup 6%).
- Genera `training_info.json` con metadatos y deja un log detallado en `logs/debug_last_run.log`.
- Ejecuta una evaluación rápida al final tomando 5 muestras del propio dataset (o un fallback predefinido) y deja la comparación esperada/obtenida en el log.

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