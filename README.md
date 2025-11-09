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
3. **Script de entrenamiento** – `scripts/finetune_lora.py` (v1.0.9) realiza duplicación inteligente del dataset y ajusta hiperparámetros para escenarios de pocos datos.
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
pip install -r requirements.txt
```
> 💡 Si partes de un servidor recién formateado, instala previamente los drivers NVIDIA, CUDA 12.1 y utilidades del sistema (detallado en secciones posteriores del repositorio original).
> 📦 El archivo `requirements.txt` (incluido en el repo) contiene todas las librerías auxiliares requeridas para el proyecto.
> 🔍 Antes de entrenar, puedes ejecutar `python scripts/validate_environment.py` para verificar versiones de Python, CUDA, VRAM disponible, dataset y dependencias.

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
- Basado en LoRA con rank 32 sobre las capas `c_attn` y `c_proj` de DialoGPT-medium.
- Duplica datasets pequeños para asegurar convergencia.
- Hiperparámetros ajustados a escenarios low-data: batch 2, 30 épocas, scheduler `cosine`, warmup 5%.
- Genera un `training_info.json` con métricas básicas y contexto de hardware.

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
│   ├── finetune_lora.py             # Entrenamiento LoRA (v1.0.9)
│   ├── inference_lora.py            # Inferencia determinista (v1.0.2)
│   └── validate_environment.py      # Checklist opcional de diagnóstico
├── .gitignore
└── README.md (este documento)
```

## 🧪 Resultados Destacados
- Con sólo 20 instrucciones iniciales se logra un chatbot que entiende flujos corporativos simples.
- El adaptador LoRA replica procedimientos secuenciales con numeraciones consistentes.
- El tiempo de entrenamiento en RTX 4060 Ti < 10 minutos.

## 🛣️ Próximos Pasos
- Aumentar el dataset con más procesos internos.
- Añadir evaluación cuantitativa (BLEU, ROUGE, precisión manual).
- Integrar despliegue vía API REST para consumir el modelo fine-tuned en producción.

## 📄 Licencia y Autor
Este proyecto es de uso personal y demuestra capacidades de MLOps / IA aplicada. Puedes reutilizarlo adaptando los scripts a tus propios datos.

**Autor:** Andrew Vergel  ·  [LinkedIn](https://www.linkedin.com/in/andrewvergel/)  ·  [Repositorio GitHub](https://github.com/andrewvergel/finetuning-linux)

¡Gracias por revisar este proyecto! Estoy abierto a colaborar en iniciativas de IA aplicada, automatización de procesos y plataformas de asistentes inteligentes.

## ☁️ Infraestructura Recomendada
Para quienes no cuenten con una RTX 4060 Ti local, recomiendo utilizar instancias bajo demanda en [cloud.vast.ai](https://cloud.vast.ai/instances/). Las pruebas finales de este proyecto se realizaron en la instancia `27712045` (host `79466`, machine `13313`) con las siguientes características:

- **GPU:** 16 GB VRAM (CUDA 12.9, ~21.6 TFLOPS)
- **CPU:** AMD Ryzen 9 3900X (12/24 hilos)
- **RAM:** 64 GB DDR4
- **Almacenamiento:** NVMe PCIe 4.0 (4 TB, ~4.7 GB/s)
- **Red:** ~1.6 Gbps simétricos

La plataforma ofrece una buena relación costo/rendimiento (≈236 DLP/$/hr) y permite desplegar rápidamente el entorno descrito en este repositorio.
