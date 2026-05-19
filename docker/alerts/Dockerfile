FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install \
    kafka-python==2.0.2 \
    psycopg[binary]==3.2.3 \
    jinja2==3.1.4

COPY src/alerts /app/src/alerts

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.alerts.notification_worker"]
