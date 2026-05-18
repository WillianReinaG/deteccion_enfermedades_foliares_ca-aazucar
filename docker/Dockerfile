# Imagen base ligera para CI (tests) y runtime Streamlit
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

COPY src ./src
COPY tests ./tests
COPY data ./data
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
# Normaliza CRLF (Windows) para evitar "no such file or directory" en Linux
RUN sed -i 's/\r$//' /app/docker/entrypoint.sh && chmod +x /app/docker/entrypoint.sh

# Artefactos y modelo se montan en runtime (no van en la imagen por tamaño)
RUN mkdir -p /app/models /app/artifacts

EXPOSE 8501

ENTRYPOINT ["/bin/sh", "/app/docker/entrypoint.sh"]
CMD ["test"]
