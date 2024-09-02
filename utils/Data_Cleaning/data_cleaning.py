import re
from pyspark.sql.types import StringType
from pyspark.sql import DataFrame
from pyspark.sql.functions import mean, col, floor, date_format, to_date, when, udf, regexp_replace, lit
from utils.logging_setup import setup_logging


# Call logging at the start of your script
logger = setup_logging("/spark-data/logs/data_cleaning.log")


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
            logger.info(f"Successfully removed duplicate rows based on columns: {columns}.")
        else:
            df = df.dropDuplicates()
            logger.info("Successfully removed duplicate rows based on all columns.")
    except Exception as e:
        logger.error(f"Error removing duplicate rows: {e}")
    
    return df

#Working with null values
def impute_nulls(df: DataFrame, columns: list, method: str, value=None) -> DataFrame:
    """
    Imputes null values and 'null' strings in specified columns using the specified method.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    columns (list): List of columns to impute.
    method (str): Imputation method ('constant', 'mean', 'median', 'mode').
    value (any, optional): Value to use if method is 'constant'.
    
    Returns:
    DataFrame: The DataFrame with null values and 'null' strings imputed.
    """
    try:
        # Replace 'null' strings with actual nulls
        for column in columns:
            df = df.withColumn(column, when(col(column).isNull() | (col(column) == 'null'), None).otherwise(col(column)))

        # Impute nulls for each column based on the specified method
        for column in columns:
            if method == 'constant':
                df = df.fillna({column: value})
                logger.info(f"Imputed nulls in column '{column}' with constant value: {value}")
            
            elif method == 'mean':
                mean_value = df.agg(mean(col(column))).collect()[0][0]
                df = df.fillna({column: mean_value})
                logger.info(f"Imputed nulls in column '{column}' with mean value: {mean_value}")
            
            elif method == 'median':
                median_value = df.approxQuantile(column, [0.5], 0.01)[0]
                df = df.fillna({column: median_value})
                logger.info(f"Imputed nulls in column '{column}' with median value: {median_value}")
            
            elif method == 'mode':
                mode_value = df.groupBy(column).count().orderBy('count', ascending=False).first()[0]
                df = df.fillna({column: mode_value})
                logger.info(f"Imputed nulls in column '{column}' with mode value: {mode_value}")
            
            else:
                raise ValueError(f"Invalid method '{method}' for column '{column}'. Options are 'constant', 'mean', 'median', 'mode'")

    except Exception as e:
        logger.error(f"Error imputing nulls in columns '{columns}' using method '{method}': {e}")
    
    return df

#Drop null values
def drop_nulls(df: DataFrame, how: str = 'any', subset: list = None) -> DataFrame:
    """
    Drops rows or columns with null values, including 'null' strings.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    how (str): Criteria for dropping ('any' or 'all').
    subset (list, optional): List of column names to consider for dropping.
    
    Returns:
    DataFrame: The DataFrame with rows or columns dropped.
    """
    # Convert 'null' strings to actual nulls
    for column in df.columns:
        df = df.withColumn(column, when(col(column).isin('null', 'None', ''), None).otherwise(col(column)))

    # Drop rows or columns with null values
    try:
        if subset:
            df = df.dropna(how=how, subset=subset)
        else:
            df = df.dropna(how=how)
        logger.info(f"Dropped rows/columns with null values using method '{how}'.")
    except Exception as e:
        logger.error(f"Error while dropping null values: {e}")
    
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
            logger.info(f"Successfully imputed missing values in column '{column}' using strategy '{strategy}'.")

    except Exception as e:
        logger.error(f"Error imputing missing values: {e}")
    
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
            logger.info(f"Successfully dropped rows/columns with missing values based on subset: {subset}.")
        else:
            df = df.dropna(how=how)
            logger.info(f"Successfully dropped rows/columns with missing values based on the '{how}' criteria.")
    except Exception as e:
        logger.error(f"Error dropping missing values: {e}")
    
    return df

#Remove decimal
def remove_decimal(df: DataFrame, columns: list) -> DataFrame:
    """
    Removes decimal values from specific columns in the DataFrame by rounding down.

    Parameters:
    df (DataFrame): The input DataFrame.
    columns (list): List of column names from which to remove decimal values.

    Returns:
    DataFrame: The DataFrame with decimal values removed from the specified columns.
    """
    if not isinstance(columns, list):
        logger.error("Parameter 'columns' must be of type 'list'.")
        return df

    for column in columns:
        if column not in df.columns:
            logger.error(f"Column '{column}' not found in the DataFrame.")
            continue  # Skip to the next column

        try:
            # Use floor to remove decimal values while keeping the original type
            df = df.withColumn(column, floor(col(column)))
            logger.info(f"Successfully removed decimal values from column '{column}'.")

        except Exception as e:
            logger.error(f"Error removing decimal values from column '{column}': {e}")

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
        logger.error(f"Column '{column}' not found in DataFrame.")
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

        logger.info(f"Successfully standardized date format in column '{column}' to '{format}'.")

    except Exception as e:
        logger.error(f"Error standardizing date format: {e}")

    return df

#Phone number handeling
def clean_phone_numbers(df: DataFrame, phone_col: str) -> DataFrame:
    """
    Cleans and formats phone numbers in the specified column of a DataFrame.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    phone_col (str): The name of the column containing phone numbers to clean.
    
    Returns:
    DataFrame: The DataFrame with cleaned and formatted phone numbers.
    """
    # Define a function to clean phone numbers using regex
    def format_phone_number(phone: str) -> str:
        try:
            # If phone is "Not Available", return it as is
            if phone == "Not Available":
                logger.info(f"Phone number '{phone}' is 'Not Available'. No changes made.")
                return phone
            
            # Remove unwanted characters and handle extensions
            phone = re.sub(r'[^\dXx]', '', phone)  # Keep only digits and X/x
            
            # Identify and separate extensions
            match = re.match(r'^(\d+)([Xx]\d+)?$', phone)
            if not match:
                logger.warning(f"Invalid phone number format detected: '{phone}'")
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
                logger.warning(f"Phone number '{phone}' is invalid after cleaning.")
                return "Invalid phone number"
            
            if extension:
                formatted_phone += f"x{extension}"
            
            # Ensure phone number starts with +1 for US numbers
            if not formatted_phone.startswith("+1"):
                formatted_phone = "+1 " + formatted_phone
            
            logger.info(f"Successfully cleaned phone number to '{formatted_phone}'")
            return formatted_phone

        except Exception as e:
            logger.error(f"Error cleaning phone number '{phone}': {e}")
            return "Invalid phone number"

    # Register the UDF
    format_phone_number_udf = udf(format_phone_number, StringType())
    
    # Apply the UDF to clean the phone numbers
    try:
        df_cleaned = df.withColumn(phone_col, format_phone_number_udf(col(phone_col)))
        logger.info(f"Phone numbers in column '{phone_col}' cleaned successfully.")
    except Exception as e:
        logger.error(f"Error applying cleaning function to column '{phone_col}': {e}")
        raise

    return df_cleaned

def handle_negative_values(df: DataFrame, columns: list, operation: str = 'absolute', apply_floor: bool = False) -> DataFrame:
    """
    Handles negative values in specific columns of the DataFrame, with options to convert, remove (set to null), 
    replace with zero, or drop rows containing negative values. Optionally, remove decimal places.

    Parameters:
    df (DataFrame): The input DataFrame.
    columns (list): List of column names to handle negative values.
    operation (str): Operation to perform on negative values. Options are 'absolute' (convert to positive), 'remove' (set to null), 
                     'zero' (set to 0), or 'drop' (remove rows with negative values).
    apply_floor (bool): Whether to apply floor to remove decimal places.

    Returns:
    DataFrame: The DataFrame with negative values handled according to the specified operation.
    """
    if operation not in ['absolute', 'remove', 'zero', 'drop']:
        logger.error("Invalid operation. Choose 'absolute', 'remove', 'zero', or 'drop'.")
        return df

    for column in columns:
        if column not in df.columns:
            logger.error(f"Column '{column}' not found in the DataFrame.")
            continue  # Skip to the next column

        try:
            if operation == 'absolute':
                # Convert negative values to positive using a conditional expression
                df = df.withColumn(column, when(col(column) < 0, -col(column)).otherwise(col(column)))
                logger.info(f"Successfully handled negative values in column '{column}' by converting to absolute values.")
            
            elif operation == 'remove':
                # Set negative values to null
                df = df.withColumn(column, when(col(column) >= 0, col(column)).otherwise(None))
                logger.info(f"Successfully handled negative values in column '{column}' by setting them to null.")
            
            elif operation == 'zero':
                # Set negative values to zero
                df = df.withColumn(column, when(col(column) >= 0, col(column)).otherwise(0))
                logger.info(f"Successfully handled negative values in column '{column}' by setting them to zero.")
            
            elif operation == 'drop':
                # Drop rows with negative values
                df = df.filter(col(column) >= 0)
                logger.info(f"Successfully handled negative values in column '{column}' by dropping rows with negative values.")
            
            if apply_floor:
                # Optionally remove decimal places by applying floor
                df = df.withColumn(column, floor(col(column)))
                logger.info(f"Applied floor operation to remove decimal places from column '{column}'.")

        except Exception as e:
            logger.error(f"Error handling negative values in column '{column}': {e}")

    return df

#remove specific strings from data
def remove_string_from_columns(df: DataFrame, columns: list, string_to_remove: str) -> DataFrame:
    """
    Removes all occurrences of a specified string from a list of columns in the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.
    columns (list): List of column names from which to remove the string.
    string_to_remove (str): The string that needs to be removed from the column values.

    Returns:
    DataFrame: The DataFrame with the specified string removed from the specified columns.
    """
    try:
        for column_name in columns:
            df = df.withColumn(column_name, regexp_replace(col(column_name), string_to_remove, " "))
            logger.info(f"Successfully removed '{string_to_remove}' from column '{column_name}'.")

    except Exception as e:
        logger.error(f"Error removing '{string_to_remove}' from columns {columns}: {e}")
    
    return df

#validate email
def validate_emails(df: DataFrame, column: str, invalid_message: str) -> DataFrame:

    """
    Validate email addresses in a specified column of the DataFrame and replace invalid emails with a custom message.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    column (str): The column containing email addresses to validate.
    invalid_message (str): Message to use for invalid email addresses.
    
    Returns:
    DataFrame: The DataFrame with invalid emails replaced by the custom message.
    """
    # Define a function to validate email addresses
    def validate_email_udf(email: str) -> str:
        """
        Validate if an email address contains '@' and '.'.
        
        :param email: The email address to validate.
        :return: The original email if valid, otherwise the invalid_message.
        """
        try:
            # Define the regex pattern for email validation
            pattern = r'^[^@]+@[^@]+\.[^@]+$'
            if re.match(pattern, email):
                return email
            else:
                return invalid_message
        except Exception as e:
            logger.error(f"Error validating email '{email}': {e}")
            return invalid_message

    # Register the UDF
    validate_email = udf(validate_email_udf, StringType())
    
    try:
        # Apply the UDF to the DataFrame
        df = df.withColumn(column, validate_email(col(column)))
        logger.info(f"Successfully validated email addresses in column '{column}'.")
    except Exception as e:
        logger.error(f"Error validating emails in column '{column}': {e}")
    
    return df

#Null counts
def count_nulls_in_column(df: DataFrame, column_name: str) -> int:
    """
    Counts the number of null values and string 'null' in a specified column of the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.
    column_name (str): The name of the column to count nulls in.

    Returns:
    int: The count of null values and string 'null' in the column.
    """
    try:
        # Count null values and string 'null'
        null_count = df.filter(
            (col(column_name).isNull()) | (col(column_name) == lit('null'))
        ).count()

        logger.info(f"Count of null values and 'null' strings in column '{column_name}': {null_count}")

    except Exception as e:
        logger.error(f"Error counting null values in column '{column_name}': {e}")
        raise

    return null_count


