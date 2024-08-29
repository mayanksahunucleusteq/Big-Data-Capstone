import logging
import os
import re
from pyspark.sql.types import StringType
from pyspark.sql import DataFrame
from pyspark.sql.functions import mean, col, floor, date_format, to_date, when, udf


#Log Directory
log_dir = '/spark-data/logs'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'data_cleaning.log')


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[
        logging.FileHandler(log_file_path), #for showing logs into file
        logging.StreamHandler() #For showing logs to console
    ])

#Removing Duplicate
def remove_duplicates(df: DataFrame, columns: list = None) -> DataFrame:
    """
    Removes duplicate rows from the DataFrame.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    columns (list, optional): List of column names to consider for identifying duplicates. If None, all columns are considered.
    
    Returns:
    DataFrame: The DataFrame with duplicates removed.
    """
    try:
        if columns:
            df = df.dropDuplicates(columns)
            logging.info(f"Successfully removed duplicate rows based on columns: {columns}.")
        else:
            df = df.dropDuplicates()
            logging.info("Successfully removed duplicate rows based on all columns.")
    except Exception as e:
        logging.error(f"Error removing duplicate rows: {e}")
    
    return df

#Working with null values
def impute_nulls(df: DataFrame, column: str, method: str, value=None) -> DataFrame:
    """
    Imputes null values in a specified column.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    column (str): The column to impute.
    method (str): Imputation method ('constant', 'mean', 'median', 'mode').
    value (any, optional): Value to use if method is 'constant'.
    
    Returns:
    DataFrame: The DataFrame with null values imputed.
    """
    if method == 'constant':
        df = df.fillna({column: value})
    elif method == 'mean':
        mean_value = df.agg(mean(col(column))).collect()[0][0]
        df = df.fillna({column: mean_value})
    elif method == 'median':
        median_value = df.approxQuantile(column, [0.5], 0.01)[0]
        df = df.fillna({column: median_value})
    elif method == 'mode':
        mode_value = df.groupBy(column).count().orderBy('count', ascending=False).first()[0]
        df = df.fillna({column: mode_value})
    else:
        raise ValueError("Invalid method. Options are 'constant', 'mean', 'median', 'mode'")
    
    return df

#Drop null values
def drop_nulls(df: DataFrame, how: str = 'any', subset: list = None) -> DataFrame:
    """
    Drops rows or columns with null values.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    how (str): Criteria for dropping ('any' or 'all').
    subset (list, optional): List of column names to consider for dropping.
    
    Returns:
    DataFrame: The DataFrame with rows or columns dropped.
    """
    if subset:
        df = df.dropna(how=how, subset=subset)
    else:
        df = df.dropna(how=how)
    
    return df

#Handel Missing Values
def impute_missing_values(df: DataFrame, strategy: str = 'mean', value=None, columns: list = None) -> DataFrame:
    """
    Imputes missing values in the DataFrame.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    strategy (str): Strategy for imputation. Options are 'mean', 'median', 'mode', or 'value'.
    value (any, optional): Value to fill if strategy is 'value'.
    columns (list, optional): List of column names to impute. If None, all columns are considered.
    
    Returns:
    DataFrame: The DataFrame with missing values imputed.
    """

    try:
        for column in columns:
            if strategy == 'mean':
                fill_value = df.agg(mean(col(column))).collect()[0][0]
            elif strategy == 'median':
                fill_value = df.approxQuantile(column, [0.5], 0.01)[0]
            elif strategy == 'mode':
                # Mode calculation
                fill_value = df.groupBy(column).count().orderBy('count', ascending=False).first()[0]
            elif strategy == 'value':
                fill_value = value
            else:
                raise ValueError("Invalid strategy. Options are 'mean', 'median', 'mode', or 'value'.")
            
            df = df.fillna({column: fill_value})
            logging.info(f"Successfully imputed missing values in column '{column}' using strategy '{strategy}'.")

    except Exception as e:
        logging.error(f"Error imputing missing values: {e}")
    
    return df

#Drop Missing Values row
def drop_missing_values(df: DataFrame, how: str = 'any', subset: list = None) -> DataFrame:
    """
    Drops rows or columns with missing values from the DataFrame.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    how (str): Criteria for dropping. Options are 'any' (drop rows/columns with any missing values) or 'all' (drop rows/columns with all missing values).
    subset (list, optional): List of column names to consider for dropping. If None, all columns are considered.
    
    Returns:
    DataFrame: The DataFrame with rows or columns dropped.
    """

    try:
        if subset:
            df = df.dropna(how=how, subset=subset)
            logging.info(f"Successfully dropped rows/columns with missing values based on subset: {subset}.")
        else:
            df = df.dropna(how=how)
            logging.info(f"Successfully dropped rows/columns with missing values based on the '{how}' criteria.")
    except Exception as e:
        logging.error(f"Error dropping missing values: {e}")
    
    return df

#Remove decimals
def remove_decimal(df: DataFrame, column: str) -> DataFrame:
    """
    Removes decimal values from a specific column in the DataFrame by rounding down.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    column (str): The column name from which to remove decimal values.
    
    Returns:
    DataFrame: The DataFrame with decimal values removed from the specified column.
    """
    if column not in df.columns:
        logging.error(f"Column '{column}' not found in the DataFrame.")
        return df

    try:
        # Use floor to remove decimal values while keeping the original type
        df = df.withColumn(column, floor(col(column)))
        logging.info(f"Successfully removed decimal values from column '{column}'.")

    except Exception as e:
        logging.error(f"Error removing decimal values from column '{column}': {e}")
    
    return df

#Date Format
def standardize_date_format(df: DataFrame, column: str, format: str = 'yyyy-MM-dd') -> DataFrame:
    """
    Standardizes the date format in a specified column to 'yyyy-MM-dd' and removes the time component.

    Parameters:
    df (DataFrame): The input DataFrame.
    column (str): The column name to standardize.
    format (str): The target date format (default is 'yyyy-MM-dd').

    Returns:
    DataFrame: The DataFrame with standardized date format in the specified column.
    """

    if column not in df.columns:
        logging.error(f"Column '{column}' not found in DataFrame.")
        return df

    try:
        # Define multiple date formats
        date_formats = [
            "dd.MM.yyyy",
            "dd/MM/yyyy",
            "dd/MMM/yyyy",
            "dd.MMMM.yyyy",
            "yyyy-MM-dd",
            "MM/dd/yyyy",
            "dd-MMM-yyyy"
        ]

        # Try parsing the date with each format
        for fmt in date_formats:
            df = df.withColumn(column, to_date(col(column), fmt))
        
        # Format the date to the standard format
        df = df.withColumn(column, date_format(col(column), format))

        # Handle cases where conversion failed
        df = df.withColumn(column, when(col(column).isNull(), 'Invalid Date').otherwise(col(column)))

        logging.info(f"Successfully standardized date format in column '{column}' to '{format}'.")

    except Exception as e:
        logging.error(f"Error standardizing date format: {e}")

    return df

#Phone number handeling
def clean_phone_numbers(df: DataFrame, phone_col: str) -> DataFrame:
    # Define a function to clean phone numbers using regex
    def format_phone_number(phone: str) -> str:
        # If phone is "Not Available", return it as is
        if phone == "Not Available":
            return phone
        
        # Remove unwanted characters and handle extensions
        phone = re.sub(r'[^\dXx]', '', phone)  # Keep only digits and X/x
        
        # Identify and separate extensions
        match = re.match(r'^(\d+)([Xx]\d+)?$', phone)
        if not match:
            return "Invalid phone number"
        
        phone, extension = match.groups()
        extension = extension[1:] if extension else ''  # Remove 'X' from extension
        
        # Normalize phone number and add +1 if not present
        if len(phone) == 10:
            formatted_phone = f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
        elif len(phone) == 11 and phone.startswith('1'):
            formatted_phone = f"+1 ({phone[1:4]}) {phone[4:7]}-{phone[7:]}"
        elif len(phone) == 12 and phone.startswith('001'):
            formatted_phone = f"+1 ({phone[3:6]}) {phone[6:9]}-{phone[9:]}"
        else:
            return "Invalid phone number"
        
        if extension:
            formatted_phone += f"x{extension}"
        
        # Ensure phone number starts with +1 for US numbers
        if not formatted_phone.startswith("+1"):
            formatted_phone = "+1 " + formatted_phone
        
        return formatted_phone

    # Register the UDF
    format_phone_number_udf = udf(format_phone_number, StringType())
    
    # Apply the UDF to clean the phone numbers
    df_cleaned = df.withColumn(phone_col, format_phone_number_udf(col(phone_col)))
    
    return df_cleaned













