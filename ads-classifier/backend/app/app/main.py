from fastapi import (
    FastAPI,
)
from app.api.v1.api import api_router as api_router_v1
from app.core.config import settings
from app import s3_client
from contextlib import asynccontextmanager
from starlette.middleware.cors import CORSMiddleware


class ExtendedFastAPI(FastAPI):
    s3_client : s3_client.S3Client

@asynccontextmanager
async def lifespan(app: ExtendedFastAPI):
    # Startup
    print("startup fastapi")

    # init s3
    app.s3_client = s3_client.S3Client(
        settings.S3_ACCESS_KEY_ID,
        settings.S3_SECRET_ACCESS_KEY,
        settings.PUBLIC_URL,
        )

    yield
    # shutdown
    print("shutdown fastapi")
    


# Core Application Instance
app = ExtendedFastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Set all CORS origins enabled
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/")
async def root():
    """
    An example "Hello world" FastAPI route.
    """
    # if oso.is_allowed(user, "read", message):
    return {"message": "Hello World"}


# Add Routers
app.include_router(api_router_v1, prefix=settings.API_V1_STR)
