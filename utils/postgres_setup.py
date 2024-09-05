from utils.logging_setup import setup_logging

# Call logger at the start of your script
logger = setup_logging("/spark-data/logs/postgres_setup.log")

#Function for save dataframe into the database
def save_dfs_to_postgres(spark, jdbc_url, properties, **dataframes):
    """
    Save multiple DataFrames to PostgreSQL, using DataFrame variable names as table names.
    :param spark: SparkSession object
    :param jdbc_url: JDBC URL for PostgreSQL
    :param properties: JDBC properties (user, password, driver)
    :param dataframes: Variable number of keyword arguments (DataFrame name: DataFrame)
    """
    for name, df in dataframes.items():
        table_name = name
        try:
            logger.info(f"Saving DataFrame to PostgreSQL table: {table_name}")
            df.write.jdbc(url=jdbc_url, table=table_name, mode="overwrite", properties=properties)
            logger.info(f"Successfully saved DataFrame to table {table_name}.")
        except Exception as e:
            logger.error(f"Error saving DataFrame to PostgreSQL table {table_name}: {e}")
            print(f"An error occurred while saving {table_name}: {e}")
            raise

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
