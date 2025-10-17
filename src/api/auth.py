# /src/api/auth.py

import logging
import os
from datetime import datetime, timedelta, timezone
from logging.config import dictConfig
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from dotenv import find_dotenv, load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel

from logging_config import LOGGING_CONFIG, ColoredFormatter

# Setup logging
dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if type(handler) is logging.StreamHandler:
        handler.setFormatter(ColoredFormatter('%(levelname)s:     %(asctime)s %(name)s - %(message)s'))


# =========================
# Конфигурация
# =========================

# .env:
env_path = find_dotenv(".env", usecwd=True)
load_dotenv(env_path, override=True, encoding="utf-8-sig")

# ADMIN_PASSWORD=super-secret
# JWT_SECRET=change-me
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
if not ADMIN_PASSWORD:
    raise RuntimeError("ADMIN_PASSWORD не задан в .env")

SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Argon2 параметры
ph = PasswordHasher(
    time_cost=3,           # итерации
    memory_cost=64 * 1024, # 64 MiB
    parallelism=2
)

# Хеш пароля администратора вычисляется один раз при старте
ADMIN_PASSWORD_HASH = ph.hash(ADMIN_PASSWORD)

# Для Swagger: tokenUrl на раут логина
# Монтируем router под /api, используем абсолютный путь.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

router = APIRouter(prefix="/auth", tags=["auth"])

# Фиксированная "карточка" пользователя admin (без БД)
APP_STARTED_AT = datetime.now(timezone.utc)
ADMIN_USER = {
    "id": 1,
    "username": ADMIN_USERNAME,
    "full_name": None,
    "role": "admin",
    "created_at": APP_STARTED_AT,
}

# =========================
# Модели
# =========================

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserRead(BaseModel):
    id: int
    username: str
    full_name: Optional[str] = None
    role: str
    created_at: datetime

# =========================
# Утилиты
# =========================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    logger.debug("Verifying password")
    try:
        return ph.verify(hashed_password, plain_password)
    except VerifyMismatchError:
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    logger.debug(f"Creating access token, expires_delta={expires_delta}")
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserRead:
    logger.debug("Getting current user from token")
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if not username or username != ADMIN_USERNAME:
            logger.warning("Invalid username in token")
            raise credentials_exception
    except JWTError:
        logger.error("JWT decode error", exc_info=True)
        raise credentials_exception

    logger.debug("Token valid, returning user info")
    return UserRead(**ADMIN_USER)

# =========================
# Эндпоинты
# =========================

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    logger.info(f"Login attempt for user: {form_data.username}")

    # Разрешаем вход только для admin
    if form_data.username != ADMIN_USERNAME:
        logger.warning(f"Unauthorized login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    if not verify_password(form_data.password, ADMIN_PASSWORD_HASH):
        logger.warning(f"Unauthorized login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    token = create_access_token(
        {"sub": ADMIN_USERNAME, "role": ADMIN_USER["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    logger.info(f"User {form_data.username} logged in successfully")
    return {"access_token": token, "token_type": "bearer"}

@router.get("/me", response_model=UserRead)
async def me(current_user: UserRead = Depends(get_current_user)):
    logger.debug("Getting current user info")
    return current_user
