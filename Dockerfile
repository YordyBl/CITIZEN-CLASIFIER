FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia e instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt -i https://pypi.org/simple

# Copia el código fuente
COPY app/ app/

# Expone el puerto de la API
EXPOSE 8000

# Ejecuta la aplicación
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
