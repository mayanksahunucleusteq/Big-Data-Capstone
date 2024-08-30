import shutil
import os
import logging
from pyspark.sql import DataFrame

#Log Directory
log_dir = '/spark-data/logs'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'save_file.log')


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[
        logging.FileHandler(log_file_path), #for showing logs into file
        logging.StreamHandler() #For showing logs to console
    ])

def save_file(df: DataFrame, output_dir: str, file_name: str, file_format: str = "csv", **options) -> str:
    """
    Save a Spark DataFrame as a single file in the specified format with error handling and logging.

    :param df: The Spark DataFrame to save.
    :param output_dir: The directory where the file will be saved.
    :param file_name: The name of the output file.
    :param file_format: The format of the output file (default is 'csv').
    :param options: Additional options for writing the file (e.g., header, delimiter).
    :return: The final output path of the saved file.
    """
    temp_dir = os.path.join(output_dir, "temp_dir")
    
    try:
        # Coalesce to a single partition to save as one file
        logging.info("Saving DataFrame to temporary directory...")
        df.coalesce(1).write.mode('overwrite').format(file_format).options(**options).save(temp_dir)
        logging.info("DataFrame saved successfully to temporary directory.")
        
        # Get the file generated in the temporary directory
        temp_file = next(file for file in os.listdir(temp_dir) if file.endswith(f'.{file_format}'))
        
        # Move and rename the file to the final output path
        final_output_path = os.path.join(output_dir, file_name + f'.{file_format}')
        shutil.move(os.path.join(temp_dir, temp_file), final_output_path)
        logging.info(f"File moved to final output path: {final_output_path}")
        
        # Remove the temporary directory
        shutil.rmtree(temp_dir)
        logging.info(f"Temporary directory removed: {temp_dir}")

        return final_output_path

    except Exception as e:
        logging.error(f"Error saving DataFrame to file: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info(f"Temporary directory removed due to error: {temp_dir}")
       
