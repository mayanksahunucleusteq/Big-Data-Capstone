import shutil
import os
from pyspark.sql import  DataFrame
from utils.logging_setup import setup_logging


# Call logging at the start of your script
logger = setup_logging("/spark-data/logs/save_file.log")

def save_file(df: DataFrame, output_dir: str, file_name: str, file_format: str = "csv", **options) -> str:
    """
    Save a Spark DataFrame as a single file in the specified format with error handling and logger.

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
        logger.info("Saving DataFrame to temporary directory...")
        df.coalesce(1).write.mode('overwrite').format(file_format).options(**options).save(temp_dir)
        logger.info("DataFrame saved successfully to temporary directory.")
        
        # Get the file generated in the temporary directory
        temp_file = next(file for file in os.listdir(temp_dir) if file.endswith(f'.{file_format}'))
        
        # Move and rename the file to the final output path
        final_output_path = os.path.join(output_dir, file_name + f'.{file_format}')
        shutil.move(os.path.join(temp_dir, temp_file), final_output_path)
        logger.info(f"File moved to final output path: {final_output_path}")
        
        # Remove the temporary directory
        shutil.rmtree(temp_dir)
        logger.info(f"Temporary directory removed: {temp_dir}")

        return final_output_path

    except Exception as e:
        logger.error(f"Error saving DataFrame to file: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Temporary directory removed due to error: {temp_dir}")
       
