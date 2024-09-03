from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, current_date, datediff, floor, coalesce, lit
from utils.logging_setup import setup_logging


# Call logging at the start of your script
logger = setup_logging("/spark-data/logs/data_transformation.log")

#Add Age column 
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
        
        logger.info(f"Column '{birth_date_column}' found. Calculating age...")

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
    

#CLV(Customer Life Time Value Calculation(Like how much he spend on our platform))
def add_clv_column(customers_df: DataFrame, orders_df: DataFrame, total_amount_column: str, customer_id_column: str = "customer_id") -> DataFrame:
    """
    Adds a 'CLV' (Customer Lifetime Value) column to the customers DataFrame by calculating 
    the sum of the total amounts from the orders DataFrame for each customer.

    :param customers_df: DataFrame containing customer information.
    :param orders_df: DataFrame containing order information.
    :param total_amount_column: The name of the column in orders_df that contains the total amount.
    :param customer_id_column: The name of the column used to join the DataFrames (default is 'customer_id').
    :return: DataFrame with the added 'CLV' column.
    """
    logger.info("Starting to add 'CLV' column to the customers DataFrame.")
    
    try:
        # Ensure the necessary columns exist in both DataFrames
        if customer_id_column not in customers_df.columns:
            logger.error(f"The specified customer ID column '{customer_id_column}' does not exist in the customers DataFrame.")
            raise ValueError(f"The specified customer ID column '{customer_id_column}' does not exist in the customers DataFrame.")
        
        if customer_id_column not in orders_df.columns or total_amount_column not in orders_df.columns:
            logger.error(f"The specified customer ID or total amount column does not exist in the orders DataFrame.")
            raise ValueError(f"The specified customer ID or total amount column does not exist in the orders DataFrame.")
        
        logger.info("Columns validated. Proceeding with CLV calculation.")
        
        # Calculate CLV by summing total amounts for each customer
        clv_df = orders_df.groupBy(customer_id_column).agg(
            F.sum(col(total_amount_column)).alias("CLV")
        )
        
        # Join the CLV back to the customers DataFrame
        customers_df = customers_df.join(clv_df, customer_id_column, "left")
        
        # Replace null values in the 'CLV' column with 0
        customers_df = customers_df.withColumn("CLV", coalesce(col("CLV"), lit(0)))
        
        logger.info("'CLV' column added successfully, with null values replaced by 0.")
        return customers_df
    
    except Exception as e:
        logger.error(f"Error adding 'CLV' column: {e}")
        raise




