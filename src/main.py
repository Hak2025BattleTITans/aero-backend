import os
import uvicorn

from api.app import app

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "2201"))

    # Для разработки можно включить hot-reload через переменную окружения
    reload_enabled = os.getenv("UVICORN_RELOAD", "false").lower() in ("1", "true", "yes")
    # В проде можно скейлить воркеры (игнорируется, если включён reload)
    workers = int(os.getenv("UVICORN_WORKERS", "1"))

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload_enabled,
        workers=(1 if reload_enabled else workers),
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_level=os.getenv("LOG_LEVEL", "info"),
    )