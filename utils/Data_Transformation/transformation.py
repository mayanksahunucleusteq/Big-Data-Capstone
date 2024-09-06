from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import to_date, col, current_date, datediff, floor, regexp_extract, col
from pyspark.sql.types import StringType
from utils.logging_setup import setup_logging


# Call logging at the start of your script
logger = setup_logging("/spark-data/logs/data_transformation.log")

#Add Age column in Customer Table
def add_age_column(df: DataFrame, birth_date_column: str) -> DataFrame:
    """
    Add an 'age' column to the DataFrame by calculating the age based on the specified birth date column.

    :param df: The Spark DataFrame to modify.
    :param birth_date_column: The name of the column containing birth dates.
    :return: The DataFrame with the added 'age' column.
    """
    logger.info("Starting to add 'age' column based on the birth date column.")

    try:
        # Ensure the birth_date_column exists in the DataFrame
        if birth_date_column not in df.columns:
            logger.error(f"The specified birth date column '{birth_date_column}' does not exist in the DataFrame.")
            raise ValueError(f"The specified birth date column '{birth_date_column}' does not exist in the DataFrame.")
        
        logger.info(f"Column '{birth_date_column}' found. Calculating age and also ensure that birth date column is date type if it is a string type then convert this into date type")

        # Check if the birth_date_column is of string type and convert it to date type if necessary
        if isinstance(df.schema[birth_date_column].dataType, StringType):
            df = df.withColumn(
                birth_date_column,
                to_date(col(birth_date_column), "yyyy-MM-dd")  # Adjust the format as needed
            )
            logger.info(f"Column '{birth_date_column}' converted to date format.")

        # Calculate age
        df = df.withColumn(
            "age",
            floor(datediff(current_date(), col(birth_date_column)) / 365.25)
        )

        logger.info("Age column added successfully.")
        return df

    except Exception as e:
        logger.error(f"Error adding age column: {e}")
        raise
    
#Add states columns to Orders Table    
def add_state_columns(df: DataFrame, shipping_col: str, billing_col: str) -> DataFrame:
    """
    Add 'shipping_state' and 'billing_state' columns to the DataFrame by extracting state abbreviations
    from the specified shipping and billing address columns.

    :param df: The Spark DataFrame to modify.
    :param shipping_col: The name of the column containing shipping addresses.
    :param billing_col: The name of the column containing billing addresses.
    :return: The DataFrame with the added 'shipping_state' and 'billing_state' columns.
    """
    logger.info("Starting to add state columns based on the specified address columns.")

    try:
        # Ensure the required columns exist in the DataFrame
        if shipping_col not in df.columns:
            logger.error(f"Column '{shipping_col}' not found in DataFrame.")
            raise ValueError(f"The DataFrame must contain '{shipping_col}' column.")
        if billing_col not in df.columns:
            logger.error(f"Column '{billing_col}' not found in DataFrame.")
            raise ValueError(f"The DataFrame must contain '{billing_col}' column.")
        
        logger.info(f"Columns '{shipping_col}' and '{billing_col}' found. Extracting state abbreviations...")

        # Extract state abbreviations from the address columns
        df = df.withColumn(
            "shipping_state",
            regexp_extract(col(shipping_col), r'\b([A-Z]{2})\b', 1)
        )
        logger.info(f"Extracted 'shipping_state' from column '{shipping_col}'.")

        df = df.withColumn(
            "billing_state",
            regexp_extract(col(billing_col), r'\b([A-Z]{2})\b', 1)
        )
        logger.info(f"Extracted 'billing_state' from column '{billing_col}'.")

        logger.info("State columns added successfully.")
        return df

    except Exception as e:
        logger.error(f"Error adding state columns: {e}")
        raise