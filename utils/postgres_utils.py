import psycopg2
from pyspark.sql.types import *
from typing import Dict, List

from utils.logging_setup import setup_logging

# Call logger at the start of your script
logger = setup_logging("/spark-data/logs/logger.log")

# Map PySpark data types to PostgreSQL data types
def map_spark_to_postgres_type(spark_type):
    try:
        if isinstance(spark_type, StringType):
            return "VARCHAR(255)"  # Use VARCHAR with a length limit for strings
        elif isinstance(spark_type, IntegerType):
            return "INTEGER"
        elif isinstance(spark_type, LongType):
            return "BIGINT"
        elif isinstance(spark_type, (DoubleType, FloatType)):
            return "FLOAT"
        elif isinstance(spark_type, BooleanType):
            return "BOOLEAN"
        elif isinstance(spark_type, TimestampType):
            return "TIMESTAMP"
        elif isinstance(spark_type, DateType):
            return "DATE"
        else:
            return "TEXT"  # Default fallback for unknown types
    except Exception as e:
        logger.error(f"Error mapping Spark type '{spark_type}' to PostgreSQL type: {e}")
        raise ValueError(f"Unhandled Spark type: {spark_type}")

#Incremental Loading
#Load all tables from DB into D/ataframe
def load_all_tables(spark, jdbc_url, properties):
    """
    Load all tables from PostgreSQL into DataFrames.
    :param spark: SparkSession object
    :param jdbc_url: JDBC URL for PostgreSQL
    :param properties: JDBC properties (user, password, driver)
    :return: Dictionary of DataFrames with table names as keys
    """
    try:
        logger.info("Fetching table names from PostgreSQL.")
        tables_query = "(SELECT table_name FROM information_schema.tables WHERE table_schema='public') as table_list"
        tables_df = spark.read.jdbc(url=jdbc_url, table=tables_query, properties=properties)
        table_names = [row['table_name'] for row in tables_df.collect()]

        dataframes = {}
        for table in table_names:
            logger.info(f"Loading table {table} into DataFrame.")
            df = spark.read.jdbc(url=jdbc_url, table=table, properties=properties)
            dataframes[table] = df

        logger.info("Successfully loaded all tables into DataFrames.")
        return dataframes

    except Exception as e:
        logger.error(f"Error loading tables from PostgreSQL: {e}")
        raise
            

#Save all the Dataframe into postgreSQL db
def save_dfs_to_postgres_upsert(spark,jdbc_url, properties, unique_key_columns: Dict[str, List[str]], **dataframes):
    """
    Save multiple DataFrames to PostgreSQL using upsert functionality or create table if it doesn't exist,
    while preserving schema types.
    
    :param jdbc_url: JDBC URL for PostgreSQL
    :param properties: JDBC properties (user, password, driver)
    :param unique_key_columns: Dictionary mapping DataFrame names to a list of unique key columns for upsert
    :param dataframes: Variable number of keyword arguments (DataFrame name: DataFrame)
    """
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=jdbc_url.split("//")[1].split(":")[0],  # Extract host from jdbc_url
        database=jdbc_url.split("/")[-1],  # Extract database name
        user=properties["user"],
        password=properties["password"]
    )
    cursor = conn.cursor()

    for table_name, df in dataframes.items():
        unique_keys = unique_key_columns.get(table_name, [])
        primary_key = unique_keys[0] if unique_keys else None  # Assume the first unique key is the primary key

        if not primary_key:
            raise ValueError(f"No primary key provided for table {table_name}.")
        
        # Check if the table exists
        check_table_query = f"""
            (SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{table_name}'
            )) AS table_exists
            """     
        # Execute the query with PySpark
        table_check_df = spark.read.jdbc(url=jdbc_url, table=check_table_query, properties=properties)

        # Check if the table exists by looking at the value in the DataFrame
        table_exists = table_check_df.collect()[0][0]

        if not table_exists:
            # Get column names and types
            column_defs = []
            for field in df.schema.fields:
                postgres_type = map_spark_to_postgres_type(field.dataType)  # Use the actual dataType, not its string representation
                column_defs.append(f"{field.name} {postgres_type}")
            
            # Create table with proper types and primary key
            create_table_query = f"""
                CREATE TABLE {table_name} (
                    {', '.join(column_defs)},
                    PRIMARY KEY ({primary_key})
                );
            """
            try:
                cursor.execute(create_table_query)
                conn.commit()
                logger.info(f"Table {table_name} created with primary key {primary_key} and proper types.")
            except Exception as e:
                logger.error(f"Error creating table {table_name}: {e}")
                conn.rollback()
                raise
        
        # Write DataFrame to a temporary PostgreSQL table
        temp_table = f"{table_name}_temp"
        try:
            df.write.jdbc(url=jdbc_url, table=temp_table, mode="overwrite", properties=properties)
        except Exception as e:
            logger.error(f"Error writing DataFrame {table_name} to PostgreSQL: {e}")
            raise

        # Prepare upsert query
        columns = ", ".join(df.columns)
        conflict_columns = primary_key
        update_columns = ", ".join([f"{col} = EXCLUDED.{col}" for col in df.columns if col != primary_key])

        upsert_query = f"""
        INSERT INTO {table_name} ({columns})
        SELECT {columns} FROM {temp_table}
        ON CONFLICT ({conflict_columns})
        DO UPDATE SET {update_columns};
        """
        
        try:
            cursor.execute(upsert_query)
            conn.commit()
            logger.info(f"DataFrame {table_name} upserted into PostgreSQL.")
        except Exception as e:
            logger.error(f"An error occurred while upserting {table_name}: {e}")
            conn.rollback()
            raise
        finally:
            # Drop the temporary table after upsert
            cursor.execute(f"DROP TABLE IF EXISTS {temp_table}")
            conn.commit()

    # Close the PostgreSQL connection
    cursor.close()
    conn.close()


