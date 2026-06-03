from app.database import get_db_connection

def log_action(user_id: int, level: str, details: str, task_id: int = None):
    """Запись системных событий и действий пользователя (Audit Trail)"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO system_logs (task_id, user_id, level, details) VALUES (%s, %s, %s, %s)",
                    (task_id, user_id, level, details)
                )
            conn.commit()
    except Exception as e:
        print(f"Logging Error: {e}")