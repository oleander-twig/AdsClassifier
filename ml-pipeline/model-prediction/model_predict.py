from config import create_config, Config
import boto3

def model_predict() -> None:
    cfg = create_config()

    s3 = boto3.client(
    cfg.s3.service_name,
    aws_access_key_id=cfg.s3.aws_access_key_id,
    aws_secret_access_key= cfg.s3.aws_secret_access_key,
    endpoint_url=cfg.s3.public_url
    )
    s3.download_file(cfg.s3.bucket_name, 'output.csv', 'output.csv')

    return 

if __name__ == "__main__":
    model_predict()