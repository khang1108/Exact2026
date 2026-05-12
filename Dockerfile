FROM python:3.12-slim-bookworm

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace:/workspace/src

COPY pyproject.toml requirements.txt README.md ./
COPY app ./app
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
