from dataclasses import dataclass
import os

@dataclass
class DbConfig:
    """конфиг для подключения к бд"""
    host: str
    port: int
    name: str
    user: str
    password: str

@dataclass
class Config:
    mode: str
    database: DbConfig


def create_config() -> Config:
    return Config(
        database=DbConfig(
            host=os.environ.get('DB_HOST'),
            port=int(os.environ.get('DB_PORT')),
            name=os.environ.get('DB_NAME'),
            user=os.environ.get('DB_USER'),
            password=os.environ.get('DB_PASSWORD')
        ),
        mode=os.environ.get('DEV')
    )
