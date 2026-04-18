FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema necessárias para pacotes compilados
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cria diretórios de dados
RUN mkdir -p data/raw data/processed models

EXPOSE 8000

# Variável de ambiente para evitar buffer de logs
ENV PYTHONUNBUFFERED=1

CMD ["python", "run_module4.py", "--host", "0.0.0.0", "--port", "8000"]
