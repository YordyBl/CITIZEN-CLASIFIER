from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

from app.model import predecir

app = FastAPI()

@app.post("/clasificar/")
async def clasificar(file: UploadFile = File(...)):
    imagen = Image.open(io.BytesIO(await file.read()))
    resultado = predecir(imagen)
    return JSONResponse(content=resultado)


