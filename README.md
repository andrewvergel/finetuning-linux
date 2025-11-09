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
- Duplica datasets pequeños hasta ~420 ejemplos solo sobre el split de entrenamiento.
- Entrenamiento balanceado: batch efectivo 8 (2×4), 28 épocas, scheduler `cosine` (warmup 12%) y sin weight decay.
- Genera `training_info.json` con metadatos y deja un log detallado en `logs/debug_last_run.log`.
- Reserva automáticamente 15% para validación, corre evaluación al final de cada época y guarda el mejor checkpoint según `eval_loss`.
- Ejecuta una evaluación rápida al final tomando 10 ejemplos del split de validación (o un fallback predefinido) y deja la comparación esperada/obtenida en el log.

### 📊 Guía rápida de hiperparámetros
- `DATASET_MIN_EXAMPLES = 300` → número mínimo de muestras tras repetir el split de entrenamiento (ej.: con 60 instrucciones reales se repite 5× hasta ~300). *Subirlo* (360) suma iteraciones cuando la pérdida sigue bajando; *bajarlo* (200) sirve para smoke-tests o datasets más ricos.
- `PER_DEVICE_BATCH_SIZE = 2` → muestras procesadas por GPU antes de acumular gradientes. Consume ~2 GB en la 4060 Ti, ideal para dejar memoria libre. *Subirlo* (4) mejora estabilidad si la VRAM lo permite; *bajarlo* (1) es la opción mínima para GPUs de 6 GB.
- `GRADIENT_ACCUMULATION = 4` → número de pasos antes de aplicar actualización (batch efectivo = 2×4 = 8). *Subirlo* (6) suaviza gradientes ruidosos; *bajarlo* (2) es útil si notas overfitting rápido.
- `NUM_EPOCHS = 28` → cada ejemplo se ve 28 veces tras repetición (~8 400 muestras). *Subirlo* (32) si la evaluación todavía mejora; *bajarlo* (20) cuando amplíes el dataset real.
- `LEARNING_RATE = 4e-5` → velocidad de aprendizaje base (40 micro). Más bajo que el default, mitiga saltos en datasets repetidos. *Subirlo* (5e-5) si la pérdida se estanca; *bajarlo* (3e-5) cuando notas oscilaciones grandes en validación.
- `WARMUP_RATIO = 0.12` → porcentaje inicial de pasos con LR creciente (primer ~1 000 pasos). *Subirlo* (0.15) si la pérdida inicial es inestable; *bajarlo* (0.08) cuando ya subiste el LR y quieres converger más rápido.
- `LORA_DROPOUT = 0.15` → regularización sobre las capas adaptadas. *Subirlo* (0.2) si persisten respuestas repetitivas; *bajarlo* (0.1) cuando incorpores más datos variados.
- `EVAL_SAMPLE_SIZE = 12` → cantidad de ejemplos del split de validación usados en la evaluación rápida. *Subirlo* (15) si agregas nuevas instrucciones y quieres más cobertura; *bajarlo* (8) para ejecuciones experimentales rápidas.

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