FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml poetry.lock README.md ./
COPY src ./src
COPY data/processed ./data/processed

RUN pip install --upgrade pip && \
    pip install .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/swiss_electricity_load/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--", "--processed-dir", "data/processed"]
