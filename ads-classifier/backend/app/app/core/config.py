import os
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl
from enum import Enum


class ModeEnum(str, Enum):
    development = "development"
    production = "production"
    testing = "testing"


class Settings(BaseSettings, extra='ignore'):
    PROJECT_NAME: str = "app"
    BACKEND_CORS_ORIGINS: list[str] | list[AnyHttpUrl]
    MODE: ModeEnum = ModeEnum.development
    API_VERSION: str = "v1"
    API_V1_STR: str = f"/api/{API_VERSION}"
    WHEATER_URL: str = "https://wttr.in"


    S3_ACCESS_KEY_ID: str = "admin"
    S3_SECRET_ACCESS_KEY: str = "asdf54321"
    PUBLIC_URL: str = "http://localhost:9000"

    
    class Config:
        case_sensitive = True
        env_file = os.path.expanduser("../../.env")


settings = Settings()
