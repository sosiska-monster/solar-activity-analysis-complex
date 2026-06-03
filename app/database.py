import os
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "postgres"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}
SECRET_TOKEN = os.getenv("API_ACCESS_TOKEN", "university-solar-key-2026")

db_pool = None

def init_db_pool():
    global db_pool
    try:
        db_pool = psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=20, **DB_CONFIG)
        print("Database connection pool created successfully")
    except Exception as e:
        print(f"Error creating DB pool: {e}")

def close_db_pool():
    global db_pool
    if db_pool:
        db_pool.closeall()
        print("Database connection pool closed")

@contextmanager
def get_db_connection():
    """Менеджер контекста для пула соединений"""
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)

def init_db_tables():
    """Создание таблиц при старте сервера"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY, username VARCHAR(100) UNIQUE NOT NULL,
                        api_token VARCHAR(255) UNIQUE NOT NULL, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cur.execute("SELECT COUNT(*) FROM users")
                if cur.fetchone()[0] == 0:
                    cur.execute("INSERT INTO users (username, api_token) VALUES (%s, %s)", ("admin", SECRET_TOKEN))
                
                cur.execute("CREATE TABLE IF NOT EXISTS tasks (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id) ON DELETE SET NULL, received_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, message TEXT)")
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS photo_results (
                        id SERIAL PRIMARY KEY, task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE,
                        file_name TEXT, status VARCHAR(50), cropped_path TEXT, final_path TEXT, mask_path TEXT,
                        sun_radius INTEGER, total_area FLOAT, wolf_number INTEGER, spots_count INTEGER,
                        groups_count INTEGER, photo_index INTEGER, photo_time TIMESTAMP WITH TIME ZONE
                    )
                """)
                cur.execute("CREATE TABLE IF NOT EXISTS spot_tracks (id SERIAL PRIMARY KEY, task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE, first_photo_id INTEGER REFERENCES photo_results(id) ON DELETE SET NULL, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP)")
                cur.execute("CREATE TABLE IF NOT EXISTS sunspots (id SERIAL PRIMARY KEY, photo_result_id INTEGER REFERENCES photo_results(id) ON DELETE CASCADE, track_id INTEGER REFERENCES spot_tracks(id) ON DELETE SET NULL, x INTEGER, y INTEGER, area FLOAT, is_umbra BOOLEAN, class VARCHAR(5))")
                cur.execute("CREATE TABLE IF NOT EXISTS spot_events (id SERIAL PRIMARY KEY, task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE, photo_result_id INTEGER REFERENCES photo_results(id) ON DELETE CASCADE, event_type VARCHAR(10), involved_tracks INTEGER[], description TEXT)")
                cur.execute("CREATE TABLE IF NOT EXISTS system_logs (id SERIAL PRIMARY KEY, task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE, user_id INTEGER REFERENCES users(id) ON DELETE SET NULL, time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, level VARCHAR(10), details TEXT)")
            conn.commit()
    except Exception as e:
        print(f"Database Init Warning: {e}")