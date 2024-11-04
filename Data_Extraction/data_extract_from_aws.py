import os
import pandas as pd
from io import BytesIO
from pyspark.sql.functions import col, explode
from pyspark.sql.types import ArrayType
from utils.logging_setup import setup_logging
import boto3


logger = setup_logging("/spark-data/logs/logger.log")

#If you want to get list of all files and folders present into the s3 call it
def list_files_in_s3(bucket_name, folder_name):
    try:
        s3 = boto3.client('s3')
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
        
        if 'Contents' not in objects:
            logger.error(f"No files found in folder {folder_name} in bucket {bucket_name}.")
            return []
        
        files = [obj['Key'] for obj in objects['Contents'] if not obj['Key'].endswith('/')]
        logger.info(f"Found {len(files)} files in folder {folder_name} in bucket {bucket_name}.")
        return files
    except Exception as e:
        logger.error(f"Error listing files in S3 folder {folder_name}: {e}")
        return []

#Return all the sheet names from specific excel file
def get_excel_sheet_names_from_s3(bucket_name, file_key):
    # Create an S3 client
    s3_client = boto3.client('s3')

    try:
        # Get the Excel file object from S3 (streaming)
        s3_object = s3_client.get_object(Bucket=bucket_name, Key=file_key)

        # Read the content of the S3 object directly into memory
        file_content = s3_object['Body'].read()

        # Load the Excel file content into a pandas ExcelFile object
        excel_file = pd.ExcelFile(BytesIO(file_content))

        # Get all sheet names
        sheet_names = excel_file.sheet_names

        return sheet_names
    except Exception as e:
        logger.error(f"Error fetching sheet names from {file_key}: {str(e)}")
        return None
    
#Data extraction ipeline from S3    
def load_files_from_s3(spark, bucket_name, folder_name=None, file_name=None):
    #Dictonary that holds all the dataframe which extract from data set
    dataframes = {}

    if not folder_name and not file_name:
        raise ValueError("You must provide either a folder_name or file_name to load files from S3.")
    
    logger.info(f"Starting to load files from bucket: {bucket_name}, folder: {folder_name}, file: {file_name}")
    
    if not file_name and folder_name:
        files_to_load = list_files_in_s3(bucket_name, folder_name)
        if not files_to_load:
            raise FileNotFoundError(f"No files found in folder {folder_name} in bucket {bucket_name}.")
    else:
        files_to_load = [os.path.join(folder_name, file_name)] if folder_name else [file_name]

    for s3_file_path in files_to_load:
        try:
            logger.info(f"Processing file: {s3_file_path}")

            #If file in Excel
            if s3_file_path.endswith(".xlsx") or s3_file_path.endswith(".xls"):
                
                sheet_names = get_excel_sheet_names_from_s3(bucket_name=bucket_name, file_key=s3_file_path)
                for sheet in sheet_names:
                    try:

                        df = spark.read.format("com.crealytics.spark.excel") \
                            .option("header", "true") \
                            .option("inferSchema", "true") \
                            .option("dataAddress", f"'{sheet}'!A1") \
                            .load(f"s3a://{bucket_name}/{s3_file_path}")
                        
                        #Check if file is not empty
                        if df.head(1):
                            df_name = f"df_{os.path.basename(s3_file_path).split('.')[0]}_{sheet}"
                            dataframes[df_name] = df
                            logger.info(f"Successfully loaded sheet '{sheet}' from Excel file: {s3_file_path} with {df.count()} rows.")
                        else:
                            logger.error(f"Failed or Skipped loaded sheet '{sheet}' from Excel file: {s3_file_path} because it is empty")

              
                    except Exception as e:
                        logger.error(f"Error reading sheet '{sheet}' in Excel file: {s3_file_path} - {e}")
                        continue

            #If File is CSV            
            elif s3_file_path.endswith(".csv"):
                try:
                    df = spark.read.format("csv") \
                    .option("header", "true") \
                    .option("inferSchema", "true") \
                    .load(f"s3a://{bucket_name}/{s3_file_path}")

                    #check if file is not empty dataframe
                    if df.head(1):
                        df_name = f"df_{os.path.basename(s3_file_path).split('.')[0]}"
                        dataframes[df_name] = df
                        logger.info(f"Loaded CSV file '{s3_file_path}' into dataframe '{df_name}' with {df.count()} rows.")  
                    else:
                        logger.error(f"Failed or Skipped loaded Csv: '{s3_file_path}' into dataframe '{df_name}' because it is empty")                  

                except Exception as e:
                    logger.error(f"Error reading file '{s3_file_path}' - {e}")


            #If file is JSON or it also accept multilined json
            elif s3_file_path.endswith(".json"):
                try:
                    s3_full_path = f"s3a://{bucket_name}/{s3_file_path}"
                    
                    # Load JSON file from S3
                    df_json = spark.read.format("json") \
                                        .option("multiline", "true") \
                                        .load(s3_full_path)
                    
                    #If dataframe is not empty
                    if df_json.head(1):
                        # Handle nested fields
                        for field in df_json.schema.fields:
                            if isinstance(field.dataType, ArrayType):
                                df_nested = df_json.withColumn(field.name, explode(col(field.name))).select(f"{field.name}.*")
                                df_name = f"df_{os.path.basename(s3_file_path).split('.')[0]}_{field.name}"
                                dataframes[df_name] = df_nested
                                logger.info(f"Loaded nested JSON field '{field.name}' from file '{s3_file_path}' into dataframe '{df_name}' with {df_nested.count()} rows.")
                            else:
                                df_name = f"df_{os.path.basename(s3_file_path).split('.')[0]}"
                                dataframes[df_name] = df_json
                                logger.info(f"Loaded JSON file '{s3_file_path}' into dataframe '{df_name}' with {df_json.count()} rows.")
                    else:
                        logger.error(f"Failed or Skipped loaded json: '{s3_file_path}' into dataframe '{df_name}' because it is empty")

                except Exception as e:
                    logger.error(f"Error processing JSON file {s3_file_path}: {str(e)}")


            #Read Parquet
            elif s3_file_path.endswith(".parquet"):
                try: 
                    df = spark.read.format("parquet").load(f"s3a://{bucket_name}/{s3_file_path}")

                    #If dataframe is not empty
                    if df.head(1):
                        df_name = f"df_{os.path.basename(s3_file_path).split('.')[0]}"
                        dataframes[df_name] = df
                        logger.info(f"Loaded Parquet file '{s3_file_path}' into dataframe '{df_name}' with {df.count()} rows.")
                    else:
                        logger.error(f"Failed or Skipped loaded parquet: '{s3_file_path}' into dataframe '{df_name}' because it is empty")

                except Exception as e:
                    logger.error(f"Error processing parquet file {s3_file_path}: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error processing file: {s3_file_path} - {e}")
            continue

    return dataframes
