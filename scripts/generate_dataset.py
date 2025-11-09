#!/usr/bin/env python3
import json
import random

# Definir templates para generar variedad
process_templates_numerado = [
    # Templates simples que solo usan {accion}
    "¿Cómo {accion}?",
    "Pasos para {accion}",
    "Dame pasos para {accion}",
    "¿Cuál es el proceso para {accion}?",
    "Instrucciones para {accion}",
    "¿Qué hago para {accion}?",
    "¿Cómo realizar {accion}?",
    "Pasos para completar {accion}",
    "Guía para {accion}",
    "Procedimiento para {accion}",
    "¿Cuál es la forma de {accion}?",
    "¿Qué pasos seguir para {accion}?",
    "Proceso paso a paso para {accion}",
    "¿Cómo se hace {accion}?",
    "¿Qué necesito para {accion}?",
]

process_templates_bullets = [
    "Bullets para {accion}",
    "Lista de verificación para {accion}",
    "¿Qué necesito para {accion}?",
    "Elementos para {accion}",
    "Puntos importantes sobre {accion}",
    "Requisitos para {accion}",
    "Checklist para {accion}",
    "Herramientas para {accion}",
    "Recursos para {accion}",
    "Criterios para {accion}",
    "Pasos resumidos de {accion}",
    "Resumen de {accion}",
    "Aspectos clave de {accion}",
    "Consideraciones para {accion}",
    "Factores importantes de {accion}",
]

# Definir acciones y procesos variados
acciones_numerado = [
    # HR
    "solicitar vacaciones", "aprobar vacaciones", "registrar empleado nuevo",
    "actualizar información personal", "cambiar datos bancarios", "consultar saldo de vacaciones",
    "solicitar capacitación", "aprobar capacitación", "registrar asistencia",
    "evaluar desempeño", "crear objetivos", "actualizar perfil",
    
    # IT
    "crear usuario nuevo", "eliminar usuario", "cambiar contraseña",
    "instalar software", "configurar email", "conectar VPN",
    "hacer backup", "restaurar datos", "actualizar sistema",
    "acceder a base de datos", "crear ticket soporte", "reportar incidente",
    
    # Finanzas
    "generar factura", "procesar pago", "conciliar cuenta bancaria",
    "crear presupuesto", "aprobar gasto", "revisar estado financiero",
    "exportar reporte", "importar datos", "cerrar período contable",
    "aprobar compra", "registrar gasto", "generar comprobante",
    
    # Operaciones
    "iniciar reunión", "programar evento", "enviar comunicación",
    "crear documento", "firmar contrato", "aprobar proyecto",
    "asignar tarea", "seguir proyecto", "completar entregable",
    "iniciar proceso", "ejecutar workflow", "validar resultado",
]

acciones_bullets = [
    "documentos requeridos", "herramientas necesarias", "personas involucradas",
    "equipos necesarios", "permisos requeridos", "escalones del proceso",
    "criterios de calidad", "medidas de seguridad", "comunicación requerida",
    "recursos disponibles", "responsabilidades", "puntos de control",
]

# Elementos para bullets
elementos_bullets = [
    "Acceso al sistema correspondiente", "Documentación oficial", "Aprobación del supervisor",
    "Certificación vigente", "Formulario completado", "Validación de datos",
    "Notificación a stakeholders", "Backup de información", "Registro de actividad",
    "Configuración de parámetros", "Pruebas de funcionamiento", "Entrega de resultados",
    "Revisión de calidad", "Comunicación de cambios", "Actualización de manuales",
    "Capacitación del usuario", "Verificación de seguridad", "Confirmación final",
    "Seguimiento post-implementación", "Monitoreo continuo", "Soporte técnico disponible",
    "Reporte de incidencias", "Escalación de problemas", "Métricas de rendimiento",
]

# Generar 500+ ejemplos
def generate_dataset():
    dataset = []
    
    # 300 ejemplos numerados
    for i in range(300):
        template = random.choice(process_templates_numerado)
        accion = random.choice(acciones_numerado)
        
        # Generar pasos consistentes para la acción
        if "solicitar" in accion:
            pasos = "1. Acceder al sistema correspondiente\n2. Completar formulario de solicitud\n3. Adjuntar documentación requerida\n4. Enviar solicitud\n5. Confirmar recepción"
        elif "aprobar" in accion:
            pasos = "1. Revisar solicitud detallada\n2. Verificar cumplimiento de requisitos\n3. Tomar decisión de aprobación\n4. Comunicar decisión al solicitante\n5. Actualizar registro del sistema"
        elif "crear" in accion:
            pasos = "1. Definir requisitos específicos\n2. Configurar parámetros necesarios\n3. Crear entrada en el sistema\n4. Verificar información ingresada\n5. Guardar y confirmar creación"
        elif "actualizar" in accion:
            pasos = "1. Localizar registro a modificar\n2. Verificar permisos de edición\n3. Realizar cambios necesarios\n4. Validar información actualizada\n5. Guardar modificaciones"
        elif "generar" in accion:
            pasos = "1. Seleccionar parámetros del reporte\n2. Ejecutar generación automática\n3. Revisar datos obtenidos\n4. Formatear documento final\n5. Distribuir a destinatarios"
        else:
            pasos = "1. Preparar recursos necesarios\n2. Ejecutar procedimiento estándar\n3. Verificar resultado obtenido\n4. Documentar actividad realizada\n5. Notificar finalización"
        
        ejemplo = {
            "system": "INSTRUCCIÓN CRÍTICA: Si la pregunta contiene 'pasos' o 'cómo', responde SIEMPRE con formato numérico 1. 2. 3. NUNCA nombres de personas.",
            "input": template.format(accion=accion),
            "output": pasos
        }
        dataset.append(ejemplo)
    
    # 200 ejemplos bullets
    for i in range(200):
        template = random.choice(process_templates_bullets)
        accion = random.choice(acciones_bullets)
        
        # Generar bullets consistentes
        elementos = random.sample(elementos_bullets, 3)
        bullets = f"• {elementos[0]}\n• {elementos[1]}\n• {elementos[2]}"
        
        ejemplo = {
            "system": "INSTRUCCIÓN CRÍTICA: Si la pregunta contiene 'bullets' o '•', responde SIEMPRE con bullets. NUNCA nombres de personas.",
            "input": template.format(accion=accion),
            "output": bullets
        }
        dataset.append(ejemplo)
    
    # Mezclar el dataset
    random.shuffle(dataset)
    return dataset

# Generar y guardar
if __name__ == "__main__":
    print("Generando dataset de 500+ instrucciones...")
    dataset = generate_dataset()
    
    with open('data/instructions.jsonl', 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Dataset generado con {len(dataset)} ejemplos")
    print(f"📊 {sum(1 for ex in dataset if 'pasos' in ex['input'] or 'cómo' in ex['input'])} ejemplos numerados")
    print(f"📊 {sum(1 for ex in dataset if 'bullets' in ex['input'] or '•' in ex['input'])} ejemplos bullets")
