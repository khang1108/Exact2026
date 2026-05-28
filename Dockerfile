FROM python:3.12-slim-bookworm

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src \
    EXACT_API_HOST=0.0.0.0 \
    EXACT_API_PORT=8080

COPY requirements-api.txt README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3).read()"

CMD ["uvicorn", "exact.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
