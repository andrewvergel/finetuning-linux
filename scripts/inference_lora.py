import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Usar GPT-2 que es compatible y el modelo que realmente entrenamos
BASE = "gpt2"
ADAPTER = "out-tinyllama-lora"

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(">> Device:", device)

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(BASE)
# Mantener modelo en CPU para generación para evitar limitaciones de MPS
model = PeftModel.from_pretrained(base, "out-tinyllama-lora")
model.eval()

def chat(user, system="Eres un asistente profesional y conciso."):
    prompt = f"System: {system}\nUser: {user}\nAssistant:"
    ids = tok(prompt, return_tensors="pt")
    # Mover inputs a CPU para evitar problemas de MPS en generación
    ids = {k: v.cpu() for k, v in ids.items()}
    
    # Generar con configuración mejorada para mejor formato
    gen = model.generate(
        **ids, 
        max_new_tokens=120,          # Más tokens para respuestas completas
        do_sample=True,              # Sampling activado
        temperature=0.6,            # Menos random para formato consistente
        top_p=0.85,                 # Nucleus sampling
        top_k=40,                   # Top-k sampling
        repetition_penalty=1.3,      # Penalizar repeticiones
        length_penalty=1.2,         # Fomentar respuestas más largas
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id
    )
    
    response = tok.decode(gen[0], skip_special_tokens=True)
    
    # Extraer solo la respuesta del assistant
    # Buscar desde "Assistant:" hasta el final
    if "Assistant:" in response:
        response = response.split("Assistant:")[1].strip()
    else:
        # Si no encuentra Assistant, tomar desde el prompt
        response = response[len(prompt):].strip()
    
    print("Respuesta del modelo:")
    print(response)
    
    # Verificar si la respuesta es válida
    if len(response) < 10:
        print("⚠️  Advertencia: Respuesta muy corta")
    if response.count("de la Cruz") > 3:
        print("⚠️  Advertencia: Detectadas repeticiones excesivas")
    if "1)" in response or "•" in response or "2)" in response:
        print("✅ Bueno: Usa el formato de pasos enseñado")
    else:
        print("⚠️  Cuidado: No sigue el formato esperado")

def test_model():
    """Función para testear la calidad del modelo fine-tuneado"""
    test_cases = [
        {
            "input": "Dame pasos para conciliar pagos de lunes",
            "expected_format": "numerado",
            "description": "Caso de pasos numerados directo"
        },
        {
            "input": "Bullets para generar reporte de ventas", 
            "expected_format": "bullets",
            "description": "Caso de bullets directo"
        },
        {
            "input": "Pasos para actualizar información personal",
            "expected_format": "numerado",
            "description": "Otro caso numerado"
        },
        {
            "input": "Bullets para consultar vacaciones",
            "expected_format": "bullets", 
            "description": "Otro caso bullets"
        }
    ]
    
    print("=== TESTING MODELO FINE-TUNEADO ===\n")
    
    scores = {
        "respuestas_coherentes": 0,
        "formato_correcto": 0,
        "total_tests": len(test_cases)
    }
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🧪 Test {i}/{len(test_cases)}: {test_case['description']}")
        print(f"❓ Input: {test_case['input']}")
        
        # Generar respuesta
        prompt = f"System: Eres un asistente profesional y conciso.\nUser: {test_case['input']}\nAssistant:"
        ids = tok(prompt, return_tensors="pt")
        ids = {k: v.cpu() for k, v in ids.items()}
        
        gen = model.generate(
            **ids, 
            max_new_tokens=80,  
            do_sample=True, 
            temperature=0.7,     
            top_p=0.9,          
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
        )
        
        response = tok.decode(gen[0], skip_special_tokens=True)
        
        # Extraer respuesta
        if "Assistant:" in response:
            response = response.split("Assistant:")[1].strip()
        else:
            response = response[len(prompt):].strip()
        
        # Validar que la respuesta no esté vacía
        if not response or len(response.strip()) < 3:
            print(f"⚠️  Advertencia: Respuesta vacía o muy corta")
            response = "Respuesta vacía del modelo"
        
        print(f"✅ Respuesta: {response}")
        
        # Validar respuesta con criterios más específicos
        words = response.split()
        if len(words) == 0:
            print("   ❌ Respuesta completamente vacía")
            continue # Skip to next test case
        
        validation_results = {
            "coherente": len(response) > 20 and not any(word in response.lower() for word in ["cruz", "juan", "español", "puede"] * 2),
            "formato_numerado": "1)" in response or "1." in response,
            "formato_bullets": "•" in response or "-" in response,
            "no_repeticiones": len(response.split()) > 0 and len(set(response.split())) / len(response.split()) > 0.5,  # Diversidad léxica
            "respuesta_completa": response.count(")") >= 2 or response.count("•") >= 2  # Al menos 2 elementos de lista
        }
        
        # Score específico por formato
        if test_case["expected_format"] == "numerado":
            if validation_results["formato_numerado"] and validation_results["respuesta_completa"]:
                scores["formato_correcto"] += 1
                print("   ✅ Formato numerado excelente")
            elif validation_results["formato_numerado"]:
                scores["formato_correcto"] += 0.5
                print("   ⚡ Formato numerado parcial")
            else:
                print("   ❌ Formato numerado no detectado")
        elif test_case["expected_format"] == "bullets":
            if validation_results["formato_bullets"] and validation_results["respuesta_completa"]:
                scores["formato_correcto"] += 1
                print("   ✅ Formato bullets excelente")
            elif validation_results["formato_bullets"]:
                scores["formato_correcto"] += 0.5
                print("   ⚡ Formato bullets parcial")
            else:
                print("   ❌ Formato bullets no detectado")
        
        # 1. Coherencia básica (longitud y contenido)
        if validation_results["coherente"] and validation_results["no_repeticiones"]:
            scores["respuestas_coherentes"] += 1
            print("   ✅ Respuesta coherente y sin repeticiones")
        else:
            print("   ❌ Respuesta incoherente o con repeticiones excesivas")
        
        print("-" * 50)
    
    # Resumen final
    print(f"\n📊 RESUMEN DE TESTING:")
    print(f"Respuestas coherentes: {scores['respuestas_coherentes']}/{scores['total_tests']} ({(scores['respuestas_coherentes']/scores['total_tests']*100):.0f}%)")
    print(f"Formato correcto: {scores['formato_correcto']}/{scores['total_tests']} ({(scores['formato_correcto']/scores['total_tests']*100):.0f}%)")
    
    # Calcular score general
    general_score = (scores['respuestas_coherentes'] + scores['formato_correcto']) / (scores['total_tests'] * 2) * 100
    print(f"Score general: {general_score:.0f}%")
    
    if general_score >= 70:
        print("🎉 ¡Buen rendimiento! El fine-tuning está funcionando.")
    elif general_score >= 40:
        print("⚡ Rendimiento moderado. Considera más datos o más épocas.")
    else:
        print("🔧 Necesita mejora. Más datos y entrenamiento son necesarios.")

if __name__ == "__main__":
    print("=== PROBANDO MODELO FINE-TUNEADO ===")
    print("Selecciona el tipo de test:")
    print("1. Test individual")
    print("2. Test completo (recomendado)")
    
    try:
        choice = input("Ingresa tu opción (1 o 2): ").strip()
        
        if choice == "2" or choice == "":
            # Test completo
            test_model()
        else:
            # Test individual
            user_input = input("Escribe tu pregunta: ")
            chat(user_input)
            
    except KeyboardInterrupt:
        print("\n\n👋 Saliendo...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
