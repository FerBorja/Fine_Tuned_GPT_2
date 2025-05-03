from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel
import torch
import gradio as gr

# --- 1. Configuración Inicial ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./english_finetuned_gpt2"  # Asegúrate que esta ruta sea correcta

# --- 2. Cargar Modelo Fine-Tuned ---
print("Cargando modelo...")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Asegurar que tenemos un token de padding

base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model = PeftModel.from_pretrained(base_model, model_path).to(device)
model.eval()  # Poner el modelo en modo evaluación

print("¡Modelo cargado correctamente!")

# --- 3. Función de Generación ---
def generate_text(
    prompt,
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2
):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Eliminar el prompt del resultado para mostrar solo lo generado
        generated_only = generated_text[len(prompt):].strip()
        
        return generated_only if generated_only else "[No additional text generated]"
    
    except Exception as e:
        return f"Error: {str(e)}"

# --- 4. Interfaz Gradio ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 Generador de Texto con GPT-2 Fine-Tuned
    Prueba tu modelo fine-tuned en inglés
    """)
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Escribe tu prompt",
                placeholder="The future of artificial intelligence...",
                lines=3
            )
            
            with gr.Accordion("Opciones Avanzadas", open=False):
                max_length = gr.Slider(50, 300, value=100, label="Longitud máxima")
                temperature = gr.Slider(0.1, 1.5, value=0.7, label="Creatividad (Temperature)")
                top_k = gr.Slider(1, 100, value=50, label="Top-K Sampling")
                top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top-P (Nucleus Sampling)")
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.2, label="Penalización de repetición")
            
            generate_btn = gr.Button("Generar Texto", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Texto Generado",
                interactive=False,
                lines=10
            )
    
    # Ejemplos predefinidos
    examples = gr.Examples(
        examples=[
            ["The impact of climate change on"],
            ["Python is a great programming language because"],
            ["Artificial intelligence will change healthcare by"],
            ["The capital of France is"],
            ["Neural networks can be used to"]
        ],
        inputs=prompt_input
    )
    
    # Evento del botón
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_length, temperature, top_k, top_p, repetition_penalty],
        outputs=output_text
    )
    
    # Evento al presionar Enter
    prompt_input.submit(
        fn=generate_text,
        inputs=[prompt_input, max_length, temperature, top_k, top_p, repetition_penalty],
        outputs=output_text
    )

# --- 5. Lanzar la Aplicación ---
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Permite acceso desde otras dispositivos en la red
        share=True,             # Cambia a True para crear un enlace público temporal
        debug=False              # Cambia a True para ver mensajes de error detallados
    )