# Используем официальный легкий образ Python
FROM python:3.10-slim

# Устанавливаем системные библиотеки, нужные для psycopg2 и SciPy
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую папку внутри контейнера
WORKDIR /app

# Копируем список зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь остальной код (main.py, .env и т.д.)
COPY . .

# Создаем папку для результатов заранее, чтобы не было ошибок прав доступа
RUN mkdir -p results

# Команда для запуска сервера FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]