import os
from pyspark.sql.functions import explode, col
from pyspark.sql.types import ArrayType
import openpyxl
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_sheet_names(file_path):
    try:
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        return workbook.sheetnames
    except Exception as e:
        logging.error(f"Error loading Excel workbook: {file_path} - {e}")
        return []

def load_files(spark):
    folder_path = "/spark-data/Data_Files"
    dataframes = {}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Handling Excel files
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                sheet_names = get_sheet_names(file_path)
                
                if not sheet_names:
                    logging.warning(f"No sheets found in Excel file: {file_path}")
                    continue

                for sheet in sheet_names:
                    try:
                        df = spark.read.format("com.crealytics.spark.excel") \
                            .option("header", "true") \
                            .option("inferSchema", "true") \
                            .option("dataAddress", f"'{sheet}'!A1") \
                            .load(file_path)
                        
                        df_name = f"df_{filename.split('.')[0]}_{sheet}"
                        dataframes[df_name] = df
                    except Exception as e:
                        logging.error(f"Error reading sheet '{sheet}' in Excel file: {file_path} - {e}")
                        continue

            # Handling CSV files
            elif filename.endswith(".csv"):
                try:
                    df = spark.read.format("csv") \
                        .option("header", "true") \
                        .option("inferSchema", "true") \
                        .load(file_path)
                    
                    df_name = f"df_{filename.split('.')[0]}"
                    dataframes[df_name] = df
                except Exception as e:
                    logging.error(f"Error reading CSV file: {file_path} - {e}")
                    continue

            # Handling JSON files
            elif filename.endswith(".json"):
                try:
                    df_json = spark.read.format("json") \
                        .option("multiline", "true") \
                        .load(file_path)
                    
                    for field in df_json.schema.fields:
                        if isinstance(field.dataType, ArrayType):
                            try:
                                df_nested = df_json.withColumn(field.name, explode(col(field.name))).select(f"{field.name}.*")
                                df_name = f"df_{filename.split('.')[0]}_{field.name}"
                                dataframes[df_name] = df_nested
                            except Exception as e:
                                logging.error(f"Error processing nested JSON field '{field.name}' in file: {file_path} - {e}")
                                continue
                        else:
                            df_name = f"df_{filename.split('.')[0]}"
                            dataframes[df_name] = df_json
                except Exception as e:
                    logging.error(f"Error reading JSON file: {file_path} - {e}")
                    continue

        except Exception as e:
            logging.error(f"Unexpected error processing file: {file_path} - {e}")
            continue

    return dataframes
