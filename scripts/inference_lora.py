#!/usr/bin/env python3
"""
LoRA Inference Script for RTX 4060 Ti
Version: 1.0.2
Author: Auto-generated from INSTRUCTIONS.md
Optimized for: RTX 4060 Ti (16GB VRAM)

Changelog:
- v1.0.2: Use deterministic decoding suited for instruction-tuned LoRA adapters
- v1.0.1: Added safety checks, improved error handling, enhanced GPU memory management
"""

import os
import json
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Version information
SCRIPT_VERSION = "1.0.2"
SCRIPT_NAME = "inference_lora.py"


def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"\'"))
    except Exception as exc:
        print(f"⚠️ Could not load {path}: {exc}")


def env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y", "on"}


load_env_file()

# Model and Paths - Match with training script
BASE_MODEL_ID = env_str("FT_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER_PATH = env_str("FT_OUT_DIR", "models/out-lora")

# Generation settings
DEFAULT_SYSTEM_PROMPT = env_str("INF_SYSTEM_PROMPT", "Eres un asistente profesional y conciso.")
DEFAULT_TEST_PROMPT = env_str(
    "INF_TEST_PROMPT", "Dame un checklist de conciliación de pagos de los lunes."
)

# Token limits - Match with training script
MAX_INPUT_TOKENS = env_int("FT_MAX_SEQ_LEN", 512)  # Match training sequence length
MAX_NEW_TOKENS = env_int("INF_MAX_NEW_TOKENS", 128)  # Reduced from 200 to match training

# Generation parameters
DO_SAMPLE = env_bool("INF_DO_SAMPLE", False)  # Use greedy decoding by default
TEMPERATURE = env_float("INF_TEMPERATURE", 0.7 if DO_SAMPLE else 1.0)
TOP_P = env_float("INF_TOP_P", 0.9 if DO_SAMPLE else 1.0)
REPETITION_PENALTY = env_float("INF_REPETITION_PENALTY", 1.05)
AUTO_START_CHAT = env_bool("INF_AUTO_CHAT", False)

# Ensure we're using the same trust settings as training
TRUST_REMOTE_CODE = env_bool("FT_TRUST_REMOTE_CODE", True)


def log_version_info():
    """Log script version and system information"""
    print(f"🤖 {SCRIPT_NAME} v{SCRIPT_VERSION}")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # System info
    print(f"💻 PyTorch: {torch.__version__}")
    print(f"🖥️ Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"📊 VRAM: {vram_gb:.1f}GB")

def get_device():
    """Auto-detect device available"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer with safety checks"""
    
    # Check if adapter exists
    if not os.path.exists(ADAPTER_PATH):
        print(f"❌ Error: Fine-tuned model not found at {ADAPTER_PATH}")
        print("Please run finetune_lora.py first to train the model.")
        return None, None, None
    
    # Check VRAM before loading
    device = get_device()
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 8:
            print(f"⚠️ Warning: Low VRAM detected ({vram_gb:.1f}GB). May encounter issues.")
        
        # Check if enough VRAM is available
        torch.cuda.empty_cache()
        available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        available_gb = available_memory / 1e9
        
        if available_gb < 6:
            print(f"⚠️ Warning: Only {available_gb:.1f}GB VRAM available. Consider closing other applications.")
    
    # Load tokenizer with same settings as training
    print(f">> Loading tokenizer from: {BASE_MODEL_ID}")
    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    except:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(">> Tokenizer loaded")
    
    # Load model with same settings as training
    print(f">> Loading base model: {BASE_MODEL_ID}")
    try:
        # Use the same device mapping as training
        device_map = {"": device.index if hasattr(device, 'index') and device.index is not None else 0}
        
        # Check if we should use 4-bit quantization (QLoRA)
        use_qlora = env_bool("FT_USE_QLORA", True)
        
        if use_qlora and 'bitsandbytes' in globals():
            print("🔧 Using QLoRA (4-bit) for inference")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                quantization_config=bnb_config,
                trust_remote_code=env_bool("FT_TRUST_REMOTE_CODE", True),
                device_map=device_map,
                torch_dtype=torch.bfloat16
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                trust_remote_code=env_bool("FT_TRUST_REMOTE_CODE", True),
                device_map=device_map,
                torch_dtype=torch.bfloat16
            )
            if not use_qlora:
                base = base.to(device)
                
        print(">> Base model loaded")
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    print(f">> Loading LoRA adapter: {ADAPTER_PATH}")
    try:
        model = PeftModel.from_pretrained(
            base, 
            ADAPTER_PATH,
            device_map={"": device},
            torch_dtype=torch.bfloat16
        )
        model.eval()
        print(">> LoRA model ready")
    except Exception as e:
        print(f"❌ Error loading LoRA adapter: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    # Memory info after loading
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        print(f"📊 GPU Memory: {memory_allocated:.1f}GB used by model")
    
    # Load training info if available
    training_info_path = os.path.join(ADAPTER_PATH, "training_info.json")
    if os.path.exists(training_info_path):
        with open(training_info_path, "r") as f:
            training_info = json.load(f)
        print(f">> Training info loaded: {training_info.get('script_version', 'unknown')} version")
    
    return model, tok, ADAPTER_PATH

def chat(user, system=DEFAULT_SYSTEM_PROMPT, model=None, tok=None, device=None):
    """Chat function with the fine-tuned model using chat template formatting"""
    if model is None or tok is None:
        print("❌ Model or tokenizer not loaded")
        return None
    
    # Check if device is available
    if device is None:
        device = get_device()
    elif device.type == "cuda" and not torch.cuda.is_available():
        print("❌ CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    
    try:
        # Format messages using the chat template
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        # Apply the chat template
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # Add assistant turn for generation
        )
        
        # Tokenize with same settings as training
        inputs = tok(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            return_token_type_ids=False
        ).to(device)
        
        # Generate response with Qwen2.5 compatible settings
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE if DO_SAMPLE else 1.0,  # Only apply temperature when sampling
                top_p=TOP_P if DO_SAMPLE else 1.0,  # Only apply top_p when sampling
                max_new_tokens=MAX_NEW_TOKENS,
                repetition_penalty=REPETITION_PENALTY,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=False,
                output_attentions=False,
                output_hidden_states=False
            )
        
        # Decode the response
        if hasattr(gen, 'sequences'):
            full_response = tok.decode(gen.sequences[0], skip_special_tokens=True)
        else:
            full_response = tok.decode(gen[0], skip_special_tokens=True)
        
        # Extract only the assistant's response (everything after the last user message)
        # This handles cases where the model might repeat the prompt
        user_content = messages[-1]["content"]
        response = full_response.split(user_content)[-1].strip()
        
        # Clean up any remaining special tokens or artifacts
        response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        
        print("\nRespuesta:", response)
        return response
        
    except Exception as e:
        print(f" Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return None

def interactive_chat(model, tok, device):
    """Interactive chat loop"""
    print("\n" + "="*50)
    print("💬 INTERACTIVE CHAT - RTX 4060 Ti")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for commands")
    print("="*50)
    
    while True:
        try:
            user_input = input("\n👤 Usuario: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'salir']:
                print("👋 ¡Hasta luego!")
                break
            elif user_input.lower() == 'help':
                print("""
📚 Comandos disponibles:
- normal text: Ask any question
- system <text>: Change system prompt
- clear: Clear conversation
- stats: Show model statistics
- quit/exit: Exit chat
                """)
                continue
            elif user_input.lower().startswith('system '):
                system_prompt = user_input[7:].strip()
                print(f"🔧 System prompt updated: {system_prompt}")
                continue
            elif user_input.lower() == 'stats':
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1e9
                    memory_cached = torch.cuda.memory_reserved() / 1e9
                    print(f"📊 GPU Memory: {memory_allocated:.1f}GB allocated, {memory_cached:.1f}GB cached")
                continue
            elif user_input.lower() == 'clear':
                print("🧹 Conversation cleared (note: model doesn't retain context)")
                continue
            elif not user_input:
                continue
                
            # Regular chat
            response = chat(user_input, model=model, tok=tok, device=device)
            
        except KeyboardInterrupt:
            print("\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

def main():
    """Main inference function"""
    # Version info
    log_version_info()
    
    # Auto-detect device
    device = get_device()
    print(f">> Device detectado: {device}")
    
    # Load model and tokenizer
    model, tok, adapter_path = load_model_and_tokenizer()
    if model is None:
        return
    
    # Load training information
    if os.path.exists(os.path.join(adapter_path, "training_info.json")):
        with open(os.path.join(adapter_path, "training_info.json"), "r") as f:
            training_info = json.load(f)
        
        print(f"\n📊 Training Summary:")
        print(f"   Model: {training_info.get('model_id', 'Unknown')}")
        print(f"   Dataset size: {training_info.get('dataset_size', 'Unknown')}")
        print(f"   Trained on: {training_info.get('training_time', 'Unknown')}")
        print(f"   Script version: {training_info.get('script_version', 'Unknown')}")
    
    # Test query
    print("\n🧪 Testing with default query...")
    test_query = DEFAULT_TEST_PROMPT
    print(f"👤 Usuario: {test_query}")
    chat(test_query, model=model, tok=tok, device=device)
    
    # Ask if user wants interactive chat
    if AUTO_START_CHAT:
        interactive_chat(model, tok, device)
        return

    print(f"\n❓ ¿Quieres iniciar chat interactivo? (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice in {'y', 'yes', 'sí', 's', ''}:
            interactive_chat(model, tok, device)
        else:
            print("👋 Exiting...")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
