from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image

# Configuración de dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar el modelo y el processor desde Hugging Face
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-7b-hf").to(device)

# Prompt complejo
prompt = (
    "Ahora te enviaré una imagen de un auto.\n\n"
    "Necesito que la analices y me devuelvas la siguiente información en este formato exacto:\n"
    "Color: <color principal del auto>\n"
    "Matrícula: <matrícula del auto, solo letras y números, sin guiones ni espacios>\n"
    "Marca: <marca del auto, por ejemplo: Toyota, Ford>\n"
    "Modelo: <modelo del auto, por ejemplo: Corolla, Fiesta>\n\n"
    "⚠️ Importante:\n"
    "- Respondé usando exactamente cuatro líneas, una para cada ítem solicitado, en el orden y formato indicados.\n"
    "- No incluyas texto adicional ni explicaciones.\n"
    "- Si algún elemento no se puede determinar con claridad, escribí: `no detectado` luego de los dos puntos."
)

# Ruta de la imagen
ruta_imagen = "img.jpg"  # Cambiá esto por la ruta real de tu imagen

# Cargar y convertir imagen
imagen = Image.open(ruta_imagen).convert("RGB")

# Procesar una sola imagen
inputs = processor(images=imagen, text=prompt, return_tensors="pt").to(device)

# Generar respuesta
outputs = model.generate(**inputs, max_new_tokens=200)
respuesta = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# Mostrar la respuesta
print("\n🧾 Descripción del auto:")
print(respuesta)
