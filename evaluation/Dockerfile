FROM python:3.11-slim

# Ustawiamy katalog roboczy
WORKDIR /app

# Zależności systemowe (pypdf/duckdb itp.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY compare_systems.py ./ 

# (opcjonalnie) jeśli chcesz kopiować .env do obrazu:
# COPY .env ./

# Domyślny command: odpal skrypt
CMD ["python", "compare_systems.py"]
