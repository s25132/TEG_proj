FROM python:3.11-slim

# Ustawiamy katalog roboczy
WORKDIR /app

# Zainstaluj systemowe zależności potrzebne m.in. przez WeasyPrint
# (cairo, pango itd.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libcairo2 \
    libpangoft2-1.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Skopiuj pliki projektu
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY generate_data.py ./ 

# (opcjonalnie) jeśli chcesz kopiować .env do obrazu:
# COPY .env ./

# Domyślny command: odpal skrypt
CMD ["python", "generate_data.py"]
