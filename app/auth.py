from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import psycopg2.extras
from app.database import get_db_connection

security = HTTPBearer()

async def verify_token(auth: HTTPAuthorizationCredentials = Security(security)):
    """Авторизация пользователя по токену через БД"""
    token = auth.credentials
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT id, username FROM users WHERE api_token = %s", (token,))
                user = cur.fetchone()
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API Token or User not found")
        return user
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail="Database Error during authentication")