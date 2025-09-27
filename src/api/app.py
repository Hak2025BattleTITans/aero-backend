import os
import logging
from logging.config import dictConfig

from fastapi import APIRouter, FastAPI, Request

from logging_config import LOGGING_CONFIG, ColoredFormatter

from api import session_router

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


@app.get("/")
async def root(request: Request):
    return {"message": "Hello World"}

for r in app.router.routes:
    if hasattr(r, "endpoint"):
        print(r.path, r.methods, r.endpoint.__module__, id(r.endpoint))