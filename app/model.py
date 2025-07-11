from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Cargar modelo y processor solo una vez
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Leer etiquetas desde archivo
with open("app/labels.txt", "r", encoding="utf-8") as f:
    etiquetas = [line.strip() for line in f if line.strip()]

# Mapa de urgencia (1 a 10)
urgencia_map = {
    "una situación de emergencia en la vía pública (persona herida, incendio, choque fuerte)": 10,
    "una falla crítica en infraestructura urbana (poste caído, semáforo roto, cable eléctrico expuesto)": 9,
    "un vehículo obstruyendo el paso peatonal o la vía": 7,
    "basura acumulada en la calle o vereda": 5,
    "bache profundo o hueco peligroso en la calzada": 7,
    "grietas en la vereda o acera rota": 4,
    "pared o estructura derrumbada en la vía pública": 8,
    "agua estancada o fuga en la calle": 6,
    "grafiti, daño visual o vandalismo en mobiliario urbano": 3,
    "objeto abandonado en la vía (colchón, mueble, escombros)": 4,
    "problema menor que no representa peligro inmediato": 2,
    "imagen irrelevante o no relacionada con problemas urbanos": 1
}

def predecir(imagen: Image.Image) -> dict:
    imagen = imagen.convert("RGB")
    inputs = processor(text=etiquetas, images=imagen, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    scores = outputs.logits_per_image.softmax(dim=1).squeeze()

    idx_max = scores.argmax().item()
    clase = etiquetas[idx_max]
    urgencia = urgencia_map.get(clase, 0)

    return {
        "clase": clase,
        "urgencia": urgencia,
        "confianza": round(scores[idx_max].item(), 3),
        "detalle": {etiquetas[i]: round(scores[i].item(), 3) for i in range(len(etiquetas))}
    }
