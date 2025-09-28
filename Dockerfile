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
    libpq-dev gcc \
&& rm -rf /var/lib/apt/lists/*

# файлы сборки


# Сначала зависимости — для лучшего layer caching
# Ожидается файл requirements.txt в корне проекта.
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -U pip setuptools wheel
RUN pip install -vv fastapi uvicorn
RUN pip install -vv -r requirements.txt

# Копируем исходники
COPY src ./src
COPY pyproject.toml setup.py ./
ENV PYTHONPATH=/app/src

# Проверим валидность pyproject.toml
RUN python - <<'PY'
from pathlib import Path
p = Path("pyproject.toml")
if p.exists():
    b = p.read_bytes()
    if b.startswith(b'\xef\xbb\xbf'):
        b = b[3:]
    b = b.replace(b'\r\n', b'\n')
    p.write_bytes(b)
    print("pyproject.toml: BOM/CRLF очищены")
PY

# Сборка .so инплейс (и диагностика, если что-то не так)
RUN set -eux; \
    python -V; \
    ls -la src || true; \
    find src -maxdepth 4 -type d -name "core*" -o -name "optim*" || true; \
    find src -maxdepth 4 -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.pyx" \) || true; \
    python setup.py build_ext --inplace -v

# Нерутовый пользователь
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5004
# Порт приложения
ENV HOST=0.0.0.0 \
    PORT=5004

# Запускаем через модуль, чтобы работал путь /src/main.py
CMD ["python", "-m", "src.main"]