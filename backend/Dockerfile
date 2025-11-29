FROM python:3.11-slim

# Katalog roboczy w kontenerze
WORKDIR /app

# Zależności systemowe (przydatne dla duckdb/pyarrow/pypdf itp.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Zależności Pythona
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiujemy kod aplikacji
COPY app/ ./app/

# Jeśli używasz .env wewnątrz kontenera i chcesz go kopiować:
# COPY .env ./

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]