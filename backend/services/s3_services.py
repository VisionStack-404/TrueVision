import boto3
from utils.config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, BUCKET_NAME

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def upload_to_s3(file_path, file_name):
    s3.upload_file(file_path, BUCKET_NAME, file_name)

    file_url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{file_name}"
    return file_url
def download_from_s3(file_name, download_path):
    s3.download_file(BUCKET_NAME, file_name, download_path)