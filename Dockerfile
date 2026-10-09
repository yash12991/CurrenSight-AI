FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CURRENSIGHT_ENABLE_EXPERIMENTAL_AUTHENTICITY=false \
    DRISHTISCAN_ENABLE_EXPERIMENTAL_AUTHENTICITY=false \
    RUPEESCAN_ENABLE_EXPERIMENTAL_AUTHENTICITY=false

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/backend/requirements.txt

COPY backend /app/backend
COPY frontend /app/frontend

RUN useradd --create-home --uid 10001 currensight \
    && chown -R currensight:currensight /app
USER currensight

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/live', timeout=3)"

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
