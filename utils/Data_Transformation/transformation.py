from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import StringType
from utils.logging_setup import setup_logging
from typing import Callable, Dict, Any


# Call logging at the start of your script
logger = setup_logging("/spark-data/logs/data_transformation.log")

# List of US state abbreviations
US_STATE_ABBREVIATIONS = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}

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
            when(
                regexp_extract(col(shipping_col), r'\b([A-Z]{2})\b', 1).isin(*US_STATE_ABBREVIATIONS.keys()),
                regexp_extract(col(shipping_col), r'\b([A-Z]{2})\b', 1)
            ).otherwise("Not Available")
        )
        logger.info(f"Extracted 'shipping_state' from column '{shipping_col}'.")

        df = df.withColumn(
            "billing_state",
            when(
                regexp_extract(col(billing_col), r'\b([A-Z]{2})\b', 1).isin(*US_STATE_ABBREVIATIONS.keys()),
                regexp_extract(col(billing_col), r'\b([A-Z]{2})\b', 1)
            ).otherwise("Not Available")
        )
        logger.info(f"Extracted 'billing_state' from column '{billing_col}'.")

        logger.info("State columns added successfully.")
        return df

    except Exception as e:
        logger.error(f"Error adding state columns: {e}")
        raise

#For wrapping all this function
def apply_transformation_step(df: DataFrame, step_func: Callable, **kwargs) -> DataFrame:
    """
    Apply a single transformation function to the DataFrame with specific parameters.

    Parameters:
    df (DataFrame): The input DataFrame.
    step_func (Callable): The transformation function to apply.
    kwargs: Additional keyword arguments for the transformation function.

    Returns:
    DataFrame: The DataFrame after applying the transformation function.
    """
    try:
        logger.info(f"Applying {step_func.__name__} with params {kwargs}")
        return step_func(df, **kwargs)
    except Exception as e:
        logger.error(f"Error while applying {step_func.__name__}: {str(e)}")
        raise

def data_transformation_pipeline(dfs: Dict[str, DataFrame], transformation_config: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, DataFrame]:
    """
    Applies a data transformation pipeline to multiple DataFrames based on configuration.

    Parameters:
    dfs (Dict[str, DataFrame]): A dictionary of DataFrames to transform, where keys are DataFrame names.
    transformation_config (Dict[str, Dict[str, Dict[str, Any]]]): A dictionary specifying transformation steps for each DataFrame.
    
    Returns:
    Dict[str, DataFrame]: A dictionary of transformed DataFrames.
    """
    transformed_dfs = {}

    for df_name, df in dfs.items():
        logger.info(f"Starting transformation process for {df_name}")
        
        if df_name in transformation_config:
            steps = transformation_config[df_name]
            for step_name, step_params in steps.items():
                try:
                    # Fetch the transformation function
                    step_func = globals().get(step_name)
                    
                    if step_func:
                        df = apply_transformation_step(df, step_func, **step_params)
                        logger.info(f"Successfully applied {step_name} to {df_name}")
                    else:
                        logger.warning(f"Transformation function {step_name} not found for {df_name}")
                except Exception as e:
                    logger.error(f"Error while applying {step_name} to {df_name}: {str(e)}", exc_info=True)
                    raise

        transformed_dfs[df_name] = df
        logger.info(f"Completed transformation process for {df_name}")

    return transformed_dfs
