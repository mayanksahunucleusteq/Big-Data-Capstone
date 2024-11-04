import logging
from utils.logging_setup import setup_logging
from snowflake.connector import connect

# Disable Snowflake logging by removing its handlers
logging.getLogger('snowflake.connector').setLevel(logging.CRITICAL)

# Call logging at the start of your script
logger = setup_logging("/spark-data/logs/logger.log")


# Function to check if a table exists in Snowflake schema
def upsert_data(df, snowflake_config, table_name, unique_key):

    logger.info(f"Starting upserting the data on: {table_name}")
    # Create a temporary table name
    try:

        logger.info(f"Creating temprory table for upserting the data on: {table_name}")
        temp_table = f"{table_name}_TEMP"

        # Write DataFrame to Snowflake temporary table (overwrite mode)
        df.write \
            .format("snowflake") \
            .options(**snowflake_config) \
            .option("dbtable", temp_table) \
            .mode("overwrite") \
            .save()
        
        logger.info(f"Sucessfully created temperory table into snowflake {temp_table}")

        # Set the database, schema, and warehouse
        database = snowflake_config['sfDatabase']  # ETL_DATA
        schema = snowflake_config['sfSchema']      # ETL_SCHEMA
        table_name = table_name.upper()

        # Construct the fully qualified table name
        table = f"{database}.{schema}.{table_name}"
        
        logger.info(f"Start Merge Operation, Upsering the data!")

        # Construct the MERGE SQL query
        merge_query = f"""
        MERGE INTO {table} AS target
        USING {temp_table} AS source
        ON target.{unique_key} = source.{unique_key}
        WHEN MATCHED THEN 
            UPDATE SET {', '.join([f'target.{col.upper()} = source.{col.upper()}' for col in df.columns])}
        WHEN NOT MATCHED THEN 
            INSERT ({', '.join([col.upper() for col in df.columns])}) 
            VALUES ({', '.join([f'source.{col.upper()}' for col in df.columns])});
        """

        # Execute the MERGE query using Snowflake connector
        try:

            # Prepare connection parameters
            connection_params = {
                'user': snowflake_config['sfUser'],
                'password': snowflake_config['sfPassword'],
                'account': snowflake_config['sfAccount'],
                'warehouse': snowflake_config['sfWarehouse'],
                'database': snowflake_config['sfDatabase'],
                'schema': snowflake_config['sfSchema'],
                'role': snowflake_config.get('sfRole')  # Optional role
            }

            with connect(**connection_params) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(merge_query)
                    logger.info("Successfully executed merge query.")
        except Exception as e:
            logger.error(f"Error executing merge query: {e}")

        logger.info(f"Drop temperory table after upserting into main table.")
        # Drop temp table after upsert using Snowflake
        drop_sql = f"DROP TABLE IF EXISTS {temp_table}"
        
        try:
            with connect(**connection_params) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(drop_sql)
                    logger.info(f"Temp table {temp_table.upper()} dropped successfully.")
        except Exception as e:
            logger.error(f"Error dropping temp table: {e}")
        
        logger.info(f"Sucessfully Upserted the dataframe into snowflake table: {table_name}")
    
    except Exception as e:
        logger.error(f"Failed to upsert error: {e}")
        raise e


# Function to check if a table exists in Snowflake schema
def table_exists(spark, snowflake_config, table_name):
    
    try:
        query = f"""
        SELECT COUNT(*) 
        FROM {snowflake_config['sfDatabase']}.INFORMATION_SCHEMA.TABLES 
        WHERE UPPER(TABLE_NAME) = UPPER('{table_name}') 
        AND UPPER(TABLE_SCHEMA) = UPPER('{snowflake_config['sfSchema']}');
        """
        logger.info(f"Checking if table exist or not in snowflake")

        try:
            result_df = spark.read \
                .format("snowflake") \
                .options(**snowflake_config) \
                .option("query", query) \
                .load()

            logger.info(f"Got success result to check if exist or not!")    
            return result_df.collect()[0][0] > 0
        except Exception as e:
            logger.error(f"Error while checking table existence: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Error Process check if table: {table_name} exist or not.")
        raise e

# Create a new Snowflake table
def create_table(df, snowflake_config, table_name):
    try:
        logger.info(f"Start Creating table into snowflake: {table_name}")

        df.write \
            .format("snowflake") \
            .options(**snowflake_config) \
            .option("dbtable", table_name) \
            .mode("overwrite") \
            .save()
        
        logger.info(f"Sucessfully Created Table into snowflake: {table_name}")

    except Exception as e:
        logger.error(f"Error Creating table: {e}")
        raise e 

# Process all dataframes dynamically from config
def process_dataframes(spark, snowflake_config, dataframes, unique_keys_config):
    try: 
        # Iterate over each dataframe in the config
        for table_name, df in dataframes.items():
            unique_key = unique_keys_config.get(table_name)

            if not unique_key:
                raise ValueError(f"Unique key for {table_name} is not found in the configuration.")

            if df is None or df.rdd.isEmpty():
                logger.warning(f"Dataframe for {table_name} is empty or not loaded correctly.")
                continue


            logger.info(f"Check if table: {table_name} is exists or not in snowflake")

            # Check if the table exists
            if table_exists(spark, snowflake_config, table_name):
                logger.info(f"Table {table_name} exists in snowflake so perform upsert for {table_name}")
                try:
                    upsert_data(df, snowflake_config, table_name, unique_key)
                    logger.info(f"Upserted Sucessfully {table_name}!")
                except Exception as e:
                    logger.error(f"Error in upserting: {e}")
            else:
                try:
                    logger.info(f"Table {table_name} does not exist. Creating table.")
                    create_table(df, snowflake_config, table_name)
                    logger.info(f"Successfully created table: {table_name}")
                except Exception as e:
                    logger.error(f"Error in Creating table in snowflake: {e}")

        logger.info(f"Sucessfully Process Dataframe: {df}")

    except Exception as e:
        logger.error(f"Error in processing Dataframe: {e}")
        raise e


