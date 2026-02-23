FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PORT=8501 \
    PROCESSED_DIR=data/processed

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["sh", "-c", "streamlit run src/swiss_electricity_load/dashboard.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true -- --processed-dir=${PROCESSED_DIR}"]
