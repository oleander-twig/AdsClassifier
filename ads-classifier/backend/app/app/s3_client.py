import boto3

class S3Client:
    def __init__(self,
                aws_access_key_id: str,
                aws_secret_access_key: str,
                endpoint_url: str,
                default_bucket_name: str = None,
                show_progress:bool = False):
        self._default_bucket_name = default_bucket_name
        self._show_progress = show_progress
        self._endpoint_url = endpoint_url
        self.session = boto3.session.Session()

        try:
            self.s3_client = self.session.client(
                service_name='s3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                endpoint_url=endpoint_url,
            )
        except Exception as e:
            raise e
        
    def download_model(self):
        self.s3_client.download_file(self._default_bucket_name, "model.pth", "/tmp/model.pth")
        self.s3_client.download_file(self._default_bucket_name, "vectorizer.bin", "/tmp/vectorizer.bin")