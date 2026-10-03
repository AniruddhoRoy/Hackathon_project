FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/hf-cache

# CPU-only torch keeps the image small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects port 7860
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
