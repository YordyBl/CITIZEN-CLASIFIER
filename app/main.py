from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from app.model import predecir

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,  
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "API de Clasificación de Imágenes"}

@app.post("/clasificar/")
async def clasificar(file: UploadFile = File(...)):
    try:
        imagen = Image.open(io.BytesIO(await file.read()))
        resultado = predecir(imagen)
        return JSONResponse(content=resultado)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Error al procesar la imagen: {str(e)}"}
        )