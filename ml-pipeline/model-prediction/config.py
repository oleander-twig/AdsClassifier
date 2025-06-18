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
class S3Config:
    """конфиг для подключения к s3 хранилищу"""
    service_name: str
    aws_access_key_id: str
    aws_secret_access_key: str
    bucket_name: str
    public_url: str

@dataclass
class Config:
    mode: str
    database: DbConfig
    s3: S3Config

def create_config() -> Config:
    return Config(
        database=DbConfig(
            host=os.environ.get('DB_HOST'),
            port=int(os.environ.get('DB_PORT')),
            name=os.environ.get('DB_NAME'),
            user=os.environ.get('DB_USER'),
            password=os.environ.get('DB_PASSWORD')
        ),
        mode=os.environ.get('DEV'),
        s3=S3Config(
            service_name=os.environ.get('SERVICE_NAME'),
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            bucket_name=os.environ.get('BUCKET_NAME'),
            public_url=os.environ.get('PUBLIC_URL')
        )
    )
