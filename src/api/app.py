import logging
import os
from logging.config import dictConfig

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api import session_router
from logging_config import LOGGING_CONFIG, ColoredFormatter

# Setup logging
dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if type(handler) is logging.StreamHandler:
        handler.setFormatter(ColoredFormatter('%(levelname)s:     %(asctime)s %(name)s - %(message)s'))

app = FastAPI()
api_v1 = APIRouter(prefix="/v1", tags=["v1"])
api_v1.include_router(session_router)

app.include_router(api_v1, prefix="/api")

origins = [
    "https://wiered.ru",
    "https://www.wiered.ru",
    "http://localhost:5173",  # твой фронтенд (Vite)
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # можно поставить ["*"] для теста
    allow_credentials=True,
    allow_methods=["*"],            # разрешить все методы (GET, POST и т.д.)
    allow_headers=["*"],            # разрешить все заголовки
)


@app.get("/")
async def root(request: Request):
    logger.info(f"Request from {request.client.host}")
    return {"message": "Hello World"}
