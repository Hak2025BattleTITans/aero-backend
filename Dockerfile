# ./Dockerfile
# Оптимально: Python 3.11 slim
ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# (опционально) системные зависимости для сборки колёс
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libpq-dev \
&& rm -rf /var/lib/apt/lists/*

# Сначала зависимости — для лучшего layer caching
# Ожидается файл requirements.txt в корне проекта.
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -U pip setuptools wheel
RUN pip install -vv fastapi uvicorn
RUN pip install -vv -r requirements.txt

# Копируем исходники
COPY src ./src
ENV PYTHONPATH=/app/src

# Нерутовый пользователь
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser


EXPOSE 5004
# Порт приложения
ENV HOST=0.0.0.0 \
    PORT=5004

# Запускаем через модуль, чтобы работал путь /src/main.py
CMD ["python", "-m", "src.main"]