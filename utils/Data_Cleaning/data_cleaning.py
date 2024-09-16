import re, os
import logging
from pyspark.sql.types import *
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from utils.logging_setup import setup_logging
from typing import Callable, Dict
from typing import Any
from fpdf import FPDF
from pyspark.sql import functions

# Call logging at the start of your script
logger = setup_logging("/spark-data/logs/data_cleaning.log")
report_folder = "/spark-data/Report"
logger.setLevel(logging.INFO)


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
def drop_nulls(df: DataFrame, columns: list = None, archive_folder: str = None, archive_df_name: str = None) -> DataFrame:
    """
    Drops rows with null values, including 'null' strings and empty strings, and archives the removed rows.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    columns (list, optional): List of column names to consider for dropping.
    archive_folder (str, optional): Path to the archive folder for saving removed rows.
    df_name (str, optional): The name of the DataFrame for archiving purposes.
    
    Returns:
    DataFrame: The DataFrame with rows dropped.
    """
    # Convert 'null', 'None', and empty strings to actual nulls
    for column in df.columns:
        df = df.withColumn(column, when(col(column).isin('null', 'None', ''), None).otherwise(col(column)))

    # Archive rows that are going to be dropped
    rows_to_remove = df.filter(df[columns[0]].isNull()) if columns else df.filter(df.isNull())

    if archive_folder and archive_df_name and not rows_to_remove.isEmpty():
        # Define the archive path and save the removed rows to CSV
        archive_path = os.path.join(archive_folder, f'df_{archive_df_name}_removed.csv')
        rows_to_remove.coalesce(1).write.csv(archive_path, header=True, mode='overwrite')
        logger.info(f"Archived removed rows to {archive_path}")

    # Drop rows with null values
    try:
        if columns:
            df_cleaned = df.dropna(subset=columns)
        else:
            df_cleaned = df.dropna()
        logger.info(f"Dropped rows with null values.")
    except Exception as e:
        logger.error(f"Error while dropping null values: {e}")
        raise

    return df_cleaned

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

        # Initialize a column with null values
        df = df.withColumn('parsed_date', lit(None).cast("date"))

        # Try parsing the date with each format in sequence
        for fmt in date_formats:
            df = df.withColumn(
                'parsed_date',
                when(col('parsed_date').isNull(), to_date(col(column), fmt)).otherwise(col('parsed_date'))
            )
        
        # Replace the original column with the parsed date
        df = df.withColumn(column, when(col('parsed_date').isNotNull(), date_format(col('parsed_date'), format))
                          .otherwise('Invalid Date'))

        # Drop the helper 'parsed_date' column
        df = df.drop('parsed_date')

        logger.info(f"Successfully standardized date format in column '{column}' to '{format}'.")

    except Exception as e:
        logger.error(f"Error standardizing date format: {e}")
        raise

    return df

#Phone number handeling
def clean_phone_numbers(df: DataFrame, column: str) -> DataFrame:
    """
    Cleans and formats phone numbers in the specified column of a DataFrame.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    column (str): The name of the column containing phone numbers to clean.
    
    Returns:
    DataFrame: The DataFrame with cleaned and formatted phone numbers.
    """
    # Define a function to clean phone numbers using regex
    def format_phone_number(phone: str) -> str:
        try:
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

        except Exception as e:
            logger.error(f"Error cleaning phone number '{phone}': {e}")
            return "Invalid phone number"

    # Register the UDF
    format_phone_number_udf = udf(format_phone_number, StringType())
    
    # Temporarily disable logging to suppress messages during DataFrame display
    original_level = logger.level
    logger.setLevel(logging.ERROR)
    
    # Apply the UDF to clean the phone numbers
    try:
        df_cleaned = df.withColumn(column, format_phone_number_udf(col(column)))
    except Exception as e:
        logger.error(f"Error applying cleaning function to column '{column}': {e}")
        raise
    finally:
        # Restore the original logging level
        logger.setLevel(original_level)
    
    return df_cleaned

#Handle negative values
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
            df = df.withColumn(column_name, regexp_replace(col(column_name), string_to_remove, ""))
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
def calculate_total_null_count(df: DataFrame) -> int:
    """
    Calculate the total number of null values (None, 'null', or literal null) in a PySpark DataFrame.

    Parameters:
    df (DataFrame): PySpark DataFrame

    Returns:
    int: Total count of null-like values across all columns
    """
    try:
        # Log the start of the null count calculation
        logger.info("Starting total null count calculation for all columns.")
        
        total_null_count = 0
        
        # Iterate through all columns to calculate total null counts
        for col in df.columns:
            null_count = df.filter((functions.col(col).isNull()) | (functions.col(col) == 'null')).count()
            total_null_count += null_count
        
        # Log the result
        logger.info(f"Total null count: {total_null_count}")
        
        return total_null_count
    
    except Exception as e:
        logger.error(f"Error occurred while calculating null counts: {e}", exc_info=True)
        return -1

#Type Conversion If needed
def convert_column_type(df: DataFrame, column: str, target_type: str) -> DataFrame:
    """
    Converts the data type of a specified column in a DataFrame with logging and error handling.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    column (str): The column name to convert.
    target_type (str): The target data type to convert the column to.
    
    Returns:
    DataFrame: The DataFrame with the specified column's type converted.
    """
    try:
        # Attempt to convert the column to the desired type
        df = df.withColumn(column, col(column).cast(target_type))
        logger.info(f"Successfully converted column '{column}' to type '{target_type}'")
    except Exception as e:
        logger.error(f"Error converting column '{column}' to type '{target_type}': {e}")
        raise ValueError(f"Failed to convert column '{column}' to type '{target_type}'") from e
    
    return df

# Wrapper Function That will help to create cleaning pipeline
def apply_cleaning_step(df: DataFrame, step_func: Callable, **kwargs) -> DataFrame:
    """
    Apply a single cleaning function to the DataFrame with specific parameters.

    Parameters:
    df (DataFrame): The input DataFrame.
    step_func (Callable): The cleaning function to apply.
    kwargs: Additional keyword arguments for the cleaning function.

    Returns:
    DataFrame: The cleaned DataFrame after applying the function, or the original DataFrame if no operation was applied.
    """
    try:
        logger.info(f"Applying {step_func.__name__} with params {kwargs}")
        
        # Apply the function only if parameters are passed, otherwise log a skip message
        if not kwargs:
            logger.warning(f"No parameters provided for {step_func.__name__}, skipping.")
            return df

        cleaned_df = step_func(df, **kwargs)
        
        if cleaned_df is not None and not cleaned_df.isEmpty():
            return cleaned_df
        else:
            logger.warning(f"Step {step_func.__name__} did not apply any changes.")
            return df
        
    except Exception as e:
        logger.error(f"Error while applying {step_func.__name__}: {str(e)}")
        raise

def data_cleaning_pipeline(dfs: Dict[str, DataFrame], cleaning_config: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, DataFrame]:
    """
    Applies a data cleaning pipeline to multiple DataFrames based on configuration and generates combined reports.

    Parameters:
    dfs (Dict[str, DataFrame]): A dictionary of DataFrames to clean, where keys are DataFrame names.
    cleaning_config (Dict[str, Dict[str, Dict[str, Any]]]): A dictionary specifying cleaning steps for each DataFrame.
    
    Returns:
    Dict[str, DataFrame]: A dictionary of cleaned DataFrames.
    """
    cleaned_dfs = {}
    all_summaries = {}  # Dictionary to store before and after summaries for all DataFrames

    for df_name, df in dfs.items():
        logger.info(f"Starting cleaning process for {df_name}")
        
        # Get summary before cleaning
        before_cleaning_summary = get_dataframe_summary(df)

        if df_name in cleaning_config:
            steps = cleaning_config[df_name]
            for step_name, step_params in steps.items():
                try:
                    # Fetch the cleaning function
                    step_func = globals().get(step_name)
                    
                    if step_func:
                        # Handle multiple sets of parameters
                        if isinstance(step_params, list):
                            for param_set in step_params:
                                df = apply_cleaning_step(df, step_func, **param_set)
                        else:
                            df = apply_cleaning_step(df, step_func, **step_params)
                    else:
                        logger.warning(f"Cleaning function {step_name} not found for {df_name}, skipping.")
                except Exception as e:
                    logger.error(f"Error while applying {step_name} to {df_name}: {str(e)}", exc_info=True)
                    raise
        else:
            logger.info(f"No cleaning steps configured for {df_name}, skipping.")
        
        # Get summary after cleaning
        after_cleaning_summary = get_dataframe_summary(df)

        # Store summaries for this DataFrame
        all_summaries[df_name] = (before_cleaning_summary, after_cleaning_summary)

        cleaned_dfs[df_name] = df
        logger.info(f"Completed cleaning process for {df_name}")

    # Generate separate reports for before and after cleaning
    before_and_after_report_path = f"{report_folder}/combined_cleaning_report.pdf"
    generate_combined_cleaning_report(all_summaries, before_and_after_report_path)
    logger.info(f"Data cleaning reports saved to {before_and_after_report_path}")

    return cleaned_dfs

#Report generation code:
def get_dataframe_summary(df: DataFrame) -> dict:
    """
    Calculate various summary metrics for the DataFrame.

    Metrics:
    - Row and column counts
    - Null counts
    - Duplicate counts
    - Data types

    Returns:
    dict: Summary metrics
    """
    # Total rows and columns
    row_count = df.count()
    column_count = len(df.columns)

    # Duplicate counts
    duplicate_count = df.count() - df.dropDuplicates().count()

    #null counts
    null_counts = calculate_total_null_count(df)

    # Data types
    column_data_types = {field.name: str(field.dataType) for field in df.schema.fields}

    summary = {
        "row_count": row_count,
        "column_count": column_count,
        "duplicate_count": duplicate_count,
        "null_counts": null_counts,
        "data_types": column_data_types
    }

    return summary

def generate_combined_cleaning_report(all_summaries: dict, output_path: str):
    """
    Generate a PDF report summarizing the cleaning process for multiple DataFrames, 
    including metrics before and after cleaning.

    Parameters:
    all_summaries (dict): A dictionary where keys are DataFrame names, and values are tuples (before_cleaning, after_cleaning)
    output_path (str): Path to save the combined PDF report
    """
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Data Cleaning Report', 0, 1, 'C')

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, title, 0, 1, 'L')
            self.ln(5)

        def chapter_body(self, body):
            self.set_font('Arial', '', 12)
            self.multi_cell(0, 10, body)
            self.ln()

        def add_table(self, table_data):
            self.set_font('Arial', 'B', 12)
            col_widths = [40, 40, 40, 40, 40]
            
            for i, header in enumerate(table_data[0]):
                self.cell(col_widths[i], 10, header, 1, 0, 'C')
            self.ln()
            
            self.set_font('Arial', '', 12)
            for row in table_data[1:]:
                for i, item in enumerate(row):
                    self.cell(col_widths[i], 10, str(item), 1, 0, 'C')
                self.ln()

    # Create instance of FPDF class
    pdf = PDF()
    pdf.add_page()
    
    # Table Headers for before cleaning
    table_data_before = [
        ["DataFrame", "Rows Before", "Columns Before", "Duplicates Before", "Null Before"]
    ]
    
    # Table Headers for after cleaning
    table_data_after = [
        ["DataFrame", "Rows After", "Columns After", "Duplicates After", "Null After"]
    ]
    
    for df_name, (before_cleaning, after_cleaning) in all_summaries.items():
        # Safely handle duplicate counts
        duplicates_before = before_cleaning.get('duplicate_count', 0)
        duplicates_after = after_cleaning.get('duplicate_count', 0)

        # Retrieve null counts before and after cleaning
        null_counts_before = before_cleaning.get('null_counts', 0)
        null_counts_after = after_cleaning.get('null_counts', 0)

        # Prepare row data for each DataFrame before cleaning
        row_before = [
            df_name,
            before_cleaning['row_count'],
            before_cleaning['column_count'],
            duplicates_before,
            null_counts_before
        ]
        table_data_before.append(row_before)

        # Prepare row data for each DataFrame after cleaning
        row_after = [
            df_name,
            after_cleaning['row_count'],
            after_cleaning['column_count'],
            duplicates_after,
            null_counts_after
        ]
        table_data_after.append(row_after)

    # Add tables to PDF
    pdf.chapter_title('Summary Before Cleaning')
    pdf.add_table(table_data_before)

    pdf.add_page()  # Add a new page for the second section

    pdf.chapter_title('Summary After Cleaning')
    pdf.add_table(table_data_after)
    
    # Save the combined PDF
    pdf.output(output_path)

