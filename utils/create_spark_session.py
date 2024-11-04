import os
import yaml
from pyspark.sql import SparkSession

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_spark_session(config):
    # Read AWS credentials from environment variables or config file
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID") or config['aws']['access_key']
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or config['aws']['secret_key']

    # Check if AWS keys are present
    if aws_access_key is None or aws_secret_key is None:
        raise ValueError("AWS access key and secret key must be set in environment variables or config file")
    
    # Read Snowflake credentials from environment variables or config file
    snowflake_user = os.getenv("SNOWFLAKE_USER") or config['snowflake']['user']
    snowflake_password = os.getenv("SNOWFLAKE_PASSWORD") or config['snowflake']['password']
    snowflake_url = config['snowflake']['url']

    # Check if Snowflake keys are present
    if snowflake_user is None or snowflake_password is None:
        raise ValueError("Snowflake user and password must be set in environment variables or config file")
    
    # Create Spark session with dynamic configuration for AWS and Google BigQuery
    spark = SparkSession.builder \
        .appName(config['spark']['app_name']) \
        .config("spark.driver.memory", config['spark']['driver_memory']) \
        .config("spark.executor.memory", config['spark']['executor_memory']) \
        .config("spark.jars.packages", ",".join(config['spark']['jars_packages']))\
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key) \
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.hadoop.fs.s3a.multipart.uploads.enabled", str(config['spark']['s3a']['multipart_uploads_enabled']).lower()) \
        .config("spark.hadoop.fs.s3a.fast.upload", str(config['spark']['s3a']['fast_upload']).lower()) \
        .config("spark.snowflake.url", snowflake_url) \
        .config("spark.snowflake.account", config['snowflake']['account']) \
        .config("spark.snowflake.user", snowflake_user) \
        .config("spark.snowflake.password", snowflake_password) \
        .getOrCreate()

    return spark
