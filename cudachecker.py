# Verificar si CUDA está disponible
import torch
print(f"¿GPU disponible?: {torch.cuda.is_available()}")
print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")

# Instalar dependencias (ejecutar en CMD/PowerShell)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# pip install transformers datasets accelerate peft wandb