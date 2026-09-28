# Runs the Gradio UI as an ASGI app. Suits any container host (Cloud Run,
# Render, Fly, Railway); those platforms hand over a port in $PORT.
FROM python:3.12-slim

# faiss and torch need libgomp at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so the layer is reused when only source changes. The CPU
# torch build is a fraction of the default wheel, which carries CUDA.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model weights are not in the repository; point this at a Hugging Face Hub id.
ENV BERT_MODEL_PATH=rajeev60/bert-mini-cyber \
    HF_HOME=/tmp/hf \
    APP_WRITABLE_DIR=/tmp \
    PORT=8080

EXPOSE 8080

# app:app is the ASGI entrypoint; launch_app() is only for running locally.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
