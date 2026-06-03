# Solar Spotter Pro 🌞

Автоматизированный программный комплекс для анализа астрономических изображений Солнца. 
Система использует методы компьютерного зрения (OpenCV) для расчета чисел Вольфа, вычисления суточной параллели и трекинга солнечных пятен с учетом дифференциального вращения.

## 🏗 Архитектура проекта
Проект спроектирован по принципам Clean Architecture и разделен на логические модули:
* `/app/api.py` — Главный оркестратор и контроллер FastAPI.
* `/app/database.py` — Управление БД (PostgreSQL Connection Pool).
* `/app/auth.py` — Защита эндпоинтов (Zero Trust, Bearer Token).
* `/app/processing/image_core.py` — Рендеринг интерфейса, динамическое масштабирование и парсинг EXIF.
* `/app/processing/sunspots.py` — Ядро компьютерного зрения: математическая морфология, кластеризация `SciPy`.
* `/app/processing/tracking.py` — Физический модуль (Модель вращения Солнца и KDTree-пространственный поиск).

## 🚀 Быстрый запуск (Docker)

1. Убедитесь, что у вас установлен **Docker Desktop** (для Windows/Mac) или `docker-compose` (для Linux).
2. Создайте файл `.env` в корне проекта (рядом с `docker-compose.yml`) со следующими переменными:
   ```env
   DB_NAME=solar_db
   DB_USER=postgres
   DB_PASS=1111
   DB_HOST=db
   DB_PORT=5432
   API_ACCESS_TOKEN=university-solar-key-2026