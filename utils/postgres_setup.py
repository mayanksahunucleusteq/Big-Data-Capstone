from utils.logging_setup import setup_logging

# Call logger at the start of your script
logger = setup_logging("/spark-data/logs/postgres_setup.log")

#Function for save dataframe into the database
import psycopg2
from pyspark.sql import DataFrame

#Save Dataframe with upsert
def save_dfs_to_postgres_upsert(spark, jdbc_url, properties, unique_key_columns, **dataframes):
    """
    Save multiple DataFrames to PostgreSQL using upsert functionality or create table if it doesn't exist.
    
    :param spark: SparkSession object
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
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{table_name}'
            );
        """)
        table_exists = cursor.fetchone()[0]

        if not table_exists:
            # Table doesn't exist, create it with primary key
            create_table_query = f"""
                CREATE TABLE {table_name} (
                    {', '.join([f"{col} TEXT" for col in df.columns])},
                    PRIMARY KEY ({primary_key})
                );
            """
            cursor.execute(create_table_query)
            conn.commit()
            print(f"Table {table_name} created with primary key {primary_key}.")
        
        # Write DataFrame to a temporary PostgreSQL table
        temp_table = f"{table_name}_temp"
        df.write.jdbc(url=jdbc_url, table=temp_table, mode="overwrite", properties=properties)
        
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
            print(f"DataFrame {table_name} upserted into PostgreSQL.")
        except Exception as e:
            print(f"An error occurred while upserting {table_name}: {e}")
            conn.rollback()
            raise
        finally:
            # Drop the temporary table after upsert
            cursor.execute(f"DROP TABLE IF EXISTS {temp_table}")
            conn.commit()

    # Close the PostgreSQL connection
    cursor.close()
    conn.close()

#Load all tables from DB into Dataframe
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
