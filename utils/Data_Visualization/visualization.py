import matplotlib.pyplot as plt
from pyspark.sql import DataFrame
from utils.logging_setup import setup_logging
from pyspark.sql.window import Window
from pyspark.sql.functions import col, sum as _sum, when, rank, round, expr, count, year, avg, floor, concat_ws

# Call logging at the start of your script
logger = setup_logging("/spark-data/logs/logger.log")

#Checking if columns exist in dataframe or not
def check_columns(df, expected_columns):
    """
    Checks if the specified columns exist in the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame.
    expected_columns (list): List of column names to check.

    Returns:
    bool: True if all expected columns are present, False otherwise.
    """
    actual_columns = df.columns
    missing_columns = [col for col in expected_columns if col not in actual_columns]
    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
    return not missing_columns

# #Top N Selling Products
def get_top_selling_products(df_order_item: DataFrame, df_product: DataFrame, product_id_col="product_id", product_name_col="product_name", quantity_col="quantity", top_n=10):
    """
    Identifies and plots the top N selling products based on total quantity sold.

    Parameters:
    df_order_item (DataFrame): The DataFrame containing order items data.
    df_product (DataFrame): The DataFrame containing product data.
    product_id_col (str): The column name for product ID in both DataFrames.
    product_name_col (str): The column name for product name in the product DataFrame.
    quantity_col (str): The column name for quantity in the order items DataFrame.
    top_n (int): The number of top selling products to retrieve and plot. Defaults to 10.

    Returns:
    None
    """
    try:
        logger.info("Starting the process to get top selling products.")

        # Check if required columns are present in both DataFrames
        if not check_columns(df_order_item, [product_id_col, quantity_col]) or not check_columns(df_product, [product_id_col, product_name_col]):
            logger.error("Required columns are missing in one of the DataFrames.")
            return

        # Join the order_item DataFrame with the product DataFrame on 'product_id'
        logger.info("Joining order_item and product DataFrames.")
        df_order_products = df_order_item.join(df_product, on=product_id_col, how='inner')

        # Group by 'product_name' and sum up the 'quantity' to get total quantities sold
        logger.info("Aggregating total quantities by product name.")
        top_products_df = df_order_products.groupBy(product_name_col).agg(
            _sum(quantity_col).alias('total_quantity')
        )

        # Sort by 'total_quantity' in descending order and select the top N products
        logger.info(f"Sorting products by total quantity and selecting the top {top_n}.")
        top_products_df = top_products_df.orderBy(col('total_quantity').desc()).limit(top_n)

        # Collect the data to a list of rows for further processing
        logger.info("Collecting the top products data.")
        top_products = top_products_df.collect()

        # Extract product names and quantities for plotting
        product_names = [row[product_name_col] for row in top_products]
        quantities = [row['total_quantity'] for row in top_products]

        logger.info("Plotting the top selling products.")
        # Plot the top N best-selling products
        plt.figure(figsize=(10, 6))
        plt.bar(product_names, quantities, color='skyblue')
        plt.title('Top Selling Products')
        plt.xlabel('Product Name')
        plt.ylabel('Quantity Sold')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

        logger.info("Successfully completed the plotting of top selling products.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

#Top N Product purchsed by age group
def get_top_products_by_age_group(df_customer: DataFrame, df_order: DataFrame, df_order_items: DataFrame, 
                                  df_product: DataFrame, customer_id_col: str, product_id_col: str, 
                                  order_id_col: str, quantity_col: str, product_name_col: str, 
                                  age_col: str = "age", top_n: int = 5):
    """
    Finds the top N products purchased by each age group.

    Parameters:
    df_customer (DataFrame): DataFrame containing customer data.
    df_order (DataFrame): DataFrame containing order data.
    df_order_items (DataFrame): DataFrame containing order item data.
    df_product (DataFrame): DataFrame containing product data.
    customer_id_col (str): Column name for customer ID.
    product_id_col (str): Column name for product ID.
    order_id_col (str): Column name for order ID.
    quantity_col (str): Column name for quantity in order items.
    product_name_col (str): Column name for product name.
    age_col (str): Column name for customer age.
    current_year (int): The current year for age calculation.
    top_n (int): The number of top products to retrieve for each age group. Default is 5.

    Returns:
    None: Displays a plot of the top N products by age group.
    """
    try:
        # Ensure required columns exist in each DataFrame
        required_customer_cols = [customer_id_col, age_col]
        required_order_cols = [customer_id_col, order_id_col]
        required_order_item_cols = [order_id_col, product_id_col, quantity_col]
        required_product_cols = [product_id_col, product_name_col]
        
        if not check_columns(df_customer, required_customer_cols) or \
           not check_columns(df_order, required_order_cols) or \
           not check_columns(df_order_items, required_order_item_cols) or \
           not check_columns(df_product, required_product_cols):
            raise ValueError("Missing required columns in one or more DataFrames.")
        
        # Step 1: Calculate age groups
        logger.info("Calculating customer age and creating age groups.")
        df_customer = df_customer.withColumn(
            "age_group", 
            when(col(age_col) < 20, "<20")
            .when((col(age_col) >= 20) & (col(age_col) < 30), "20-29")
            .when((col(age_col) >= 30) & (col(age_col) < 40), "30-39")
            .when((col(age_col) >= 40) & (col(age_col) < 50), "40-49")
            .otherwise("50+")
        )

        # Step 2: Join the tables: customers, orders, and order items
        logger.info("Joining customer, order, and order item DataFrames.")
        df_combined = df_customer.join(df_order, customer_id_col) \
                                 .join(df_order_items, order_id_col)

        # Step 3: Group by age_group and product_id, sum the quantity for each group
        logger.info("Aggregating product purchases by age group and product.")
        df_age_group_product_count = df_combined.groupBy("age_group", product_id_col) \
                                                .agg(_sum(quantity_col).alias("total_quantity"))

        # Step 4: Find the top N products for each age group using a window specification
        logger.info(f"Retrieving the top {top_n} products for each age group.")
        window_spec = Window.partitionBy("age_group").orderBy(col("total_quantity").desc())
        df_top_products = df_age_group_product_count.withColumn("rank", rank().over(window_spec)) \
                                                    .filter(col("rank") <= top_n)

        # Step 5: Join with product names
        logger.info("Joining with product names for top products.")
        df_top_products = df_top_products.join(df_product, product_id_col, "inner")

        # Step 6: Collect data for plotting
        logger.info("Collecting data for plotting.")
        top_products_data = df_top_products.collect()

        # Prepare data for plotting
        age_groups = [row['age_group'] for row in top_products_data]
        product_names = [row[product_name_col] for row in top_products_data]
        quantities = [row['total_quantity'] for row in top_products_data]

        # Step 7: Plotting the top N products by age group
        logger.info(f"Plotting the top {top_n} products by age group.")
        plt.figure(figsize=(12, 8))
        for age_group in set(age_groups):
            idx = [i for i, group in enumerate(age_groups) if group == age_group]
            plt.barh([product_names[i] for i in idx], [quantities[i] for i in idx], label=age_group)

        plt.title(f"Top {top_n} Products by Age Group")
        plt.xlabel("Total Quantity Sold")
        plt.ylabel("Product Name")
        plt.legend(title="Age Group")
        plt.tight_layout()
        plt.show()

        logger.info(f"Successfully completed the top {top_n} products by age group plotting.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

#Top N Revenue generating products
def get_top_revenue_generating_products(df_order_items: DataFrame, df_product: DataFrame, product_id_col: str, 
                                        quantity_col: str, price_col: str, product_name_col: str, top_n: int = 10):
    """
    Finds the top N highest revenue-generating products.

    Parameters:
    df_order_items (DataFrame): DataFrame containing order items data.
    df_product (DataFrame): DataFrame containing product data.
    product_id_col (str): Column name for product ID.
    quantity_col (str): Column name for quantity in order items.
    price_col (str): Column name for product price.
    product_name_col (str): Column name for product name.
    top_n (int): The number of top products to retrieve by revenue. Default is 10.

    Returns:
    None: Displays a plot of the top N highest revenue-generating products.
    """
    try:
        # Ensure required columns exist
        required_order_item_cols = [product_id_col, quantity_col]
        required_product_cols = [product_id_col, price_col, product_name_col]
        
        if not check_columns(df_order_items, required_order_item_cols) or \
           not check_columns(df_product, required_product_cols):
            raise ValueError("Missing required columns in one or more DataFrames.")
        
        # Step 1: Join the order items DataFrame with the products DataFrame on 'product_id'
        logger.info("Joining order items and product DataFrames.")
        df_order_items_products = df_order_items.alias("oi").join(
            df_product.alias("p"), on=col(f"oi.{product_id_col}") == col(f"p.{product_id_col}"), how='inner'
        )
        
        # Step 2: Calculate the revenue for each product (quantity * product price)
        logger.info("Calculating revenue for each product.")
        df_order_items_products = df_order_items_products.withColumn(
            "revenue", col(f"oi.{quantity_col}") * col(f"p.{price_col}")
        )

        # Step 3: Group by 'product_name' and sum up the 'revenue'
        logger.info("Aggregating total revenue by product name.")
        top_revenue_products_df = df_order_items_products.groupBy(col(f"p.{product_name_col}").alias("product_name")) \
                                                         .agg(round(_sum("revenue"), 2).alias("total_revenue"))

        # Step 4: Sort by 'total_revenue' in descending order and select the top N
        logger.info(f"Sorting products by total revenue and selecting the top {top_n}.")
        top_revenue_products_df = top_revenue_products_df.orderBy(col("total_revenue").desc()).limit(top_n)

        # Step 5: Collect the data for plotting
        logger.info("Collecting the top products by revenue data.")
        top_revenue_products = top_revenue_products_df.collect()

        # Extract product names and revenues for plotting
        product_names = [row["product_name"] for row in top_revenue_products]
        revenues = [row['total_revenue'] for row in top_revenue_products]

        logger.info("Plotting the top revenue-generating products.")
        # Step 6: Plot the top N highest revenue-generating products
        plt.figure(figsize=(12, 8))
        plt.barh(product_names, revenues, color='seagreen')
        plt.title(f"Top {top_n} Highest Revenue Generating Products")
        plt.xlabel("Total Revenue")
        plt.ylabel("Product Name")
        plt.tight_layout()
        plt.show()

        logger.info(f"Successfully completed the top {top_n} highest revenue-generating products plotting.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

#Frequent Ordered Porducts
def get_frequently_ordered_products(df_order_items: DataFrame, df_product: DataFrame, product_id_col: str, 
                                    product_name_col: str, top_n: int = 10):
    """
    Finds the top N most frequently ordered products.

    Parameters:
    df_order_items (DataFrame): DataFrame containing order items data.
    df_product (DataFrame): DataFrame containing product data.
    product_id_col (str): Column name for product ID.
    product_name_col (str): Column name for product name.
    top_n (int): The number of top products to retrieve by order frequency. Default is 10.

    Returns:
    None: Displays a plot of the top N most frequently ordered products.
    """
    try:
        logger.info("Starting the process to calculate the most frequently ordered products.")

        # Ensure required columns exist
        required_order_item_cols = [product_id_col]
        required_product_cols = [product_id_col, product_name_col]
        
        if not check_columns(df_order_items, required_order_item_cols) or \
           not check_columns(df_product, required_product_cols):
            raise ValueError("Missing required columns in one or more DataFrames.")

        # Step 1: Join the order items DataFrame with the products DataFrame on 'product_id'
        logger.info("Joining order items and product DataFrames.")
        df_order_items_products = df_order_items.alias("oi").join(
            df_product.alias("p"), on=col(f"oi.{product_id_col}") == col(f"p.{product_id_col}"), how='inner'
        )

        # Step 2: Group by 'product_name' and count the number of orders
        logger.info("Aggregating the number of times each product was ordered.")
        top_frequent_products_df = df_order_items_products.groupBy(col(f"p.{product_name_col}").alias("product_name")) \
                                                          .agg(count(f"oi.{product_id_col}").alias("order_count"))

        # Step 3: Sort by 'order_count' in descending order and select the top N
        logger.info(f"Sorting products by order count and selecting the top {top_n}.")
        top_frequent_products_df = top_frequent_products_df.orderBy(col("order_count").desc()).limit(top_n)

        # Step 4: Collect the data for plotting
        logger.info("Collecting the top frequently ordered products data.")
        top_frequent_products = top_frequent_products_df.collect()

        # Extract product names and order counts for plotting
        product_names = [row["product_name"] for row in top_frequent_products]
        order_counts = [row["order_count"] for row in top_frequent_products]

        logger.info("Plotting the top frequently ordered products.")
        # Step 5: Plot the top N most frequently ordered products
        plt.figure(figsize=(12, 8))
        plt.barh(product_names, order_counts, color='coral')
        plt.title(f"Top {top_n} Most Frequently Ordered Products")
        plt.xlabel("Order Count")
        plt.ylabel("Product Name")
        plt.tight_layout()
        plt.show()

        logger.info(f"Successfully completed the top {top_n} frequently ordered products plotting.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

#Frequent Ordered Products Yearly
def get_frequently_ordered_products_by_year(df_order_items: DataFrame, df_product: DataFrame, df_order: DataFrame, 
                                             order_id_col: str, product_id_col: str, product_name_col: str, 
                                             order_date_col: str, year_filter: int, top_n: int = 10):
    """
    Finds the top N most frequently ordered products for a specific year.

    Parameters:
    df_order_items (DataFrame): DataFrame containing order items data.
    df_product (DataFrame): DataFrame containing product data.
    df_order (DataFrame): DataFrame containing order data.
    order_id_col (str): Column name for order ID in the order items and orders DataFrames.
    product_id_col (str): Column name for product ID in the order items and products DataFrames.
    product_name_col (str): Column name for product name in the products DataFrame.
    order_date_col (str): Column name for order date in the orders DataFrame.
    year_filter (int): The year for which to filter the orders.
    top_n (int): The number of top products to retrieve by order frequency. Default is 10.

    Returns:
    None: Displays a plot of the top N most frequently ordered products for the given year.
    """
    try:
        logger.info(f"Starting the process to calculate the most frequently ordered products for the year {year_filter}.")

        # Check if required columns are present
        if not check_columns(df_order_items, [order_id_col, product_id_col]) or \
           not check_columns(df_product, [product_id_col, product_name_col]) or \
           not check_columns(df_order, [order_id_col, order_date_col]):
            logger.error("Required columns are missing in one of the DataFrames.")
            return

        # Step 1: Filter the orders for the given year
        logger.info(f"Filtering orders for the year {year_filter}.")
        df_filtered_orders = df_order.filter(year(col(order_date_col)) == year_filter)

        # Step 2: Join the filtered orders with the order items and products
        logger.info("Joining filtered orders, order items, and product DataFrames.")
        df_order_items_products = df_order_items.join(df_filtered_orders, on=order_id_col, how='inner') \
                                                .join(df_product, on=product_id_col, how='inner')

        # Step 3: Group by product name and count the number of orders
        logger.info("Aggregating the number of times each product was ordered.")
        top_frequent_products_df = df_order_items_products.groupBy(col(product_name_col)) \
                                                          .agg(count(product_id_col).alias("order_count"))

        # Step 4: Sort by order count in descending order and select the top N
        logger.info(f"Sorting products by order count and selecting the top {top_n}.")
        top_frequent_products_df = top_frequent_products_df.orderBy(col("order_count").desc()).limit(top_n)

        # Step 5: Collect the data to a list of rows for plotting
        logger.info("Collecting the top frequently ordered products data.")
        top_frequent_products = top_frequent_products_df.collect()

        # Extract product names and order counts for plotting
        product_names = [row[product_name_col] for row in top_frequent_products]
        order_counts = [row['order_count'] for row in top_frequent_products]

        logger.info("Plotting the top frequently ordered products for the year.")
        # Step 6: Plot the top N most frequently ordered products
        plt.figure(figsize=(12, 8))
        plt.barh(product_names, order_counts, color='coral')
        plt.title(f"Top {top_n} Most Frequently Ordered Products in {year_filter}")
        plt.xlabel("Order Count")
        plt.ylabel("Product Name")
        plt.tight_layout()
        plt.show()

        logger.info(f"Successfully completed the top {top_n} frequently ordered products for the year {year_filter}.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


#Frequent Ordered Product by Age_Groups
def get_frequently_ordered_products_by_age_group(df_order_items: DataFrame, df_product: DataFrame, df_order: DataFrame, 
                                                 df_customer: DataFrame, order_id_col: str, product_id_col: str, 
                                                 product_name_col: str, customer_id_col: str, order_date_col: str, 
                                                 birth_date_col: str, top_n: int = 5):
    """
    Finds the top N most frequently ordered products for each customer age group.

    Parameters:
    df_order_items (DataFrame): DataFrame containing order items data.
    df_product (DataFrame): DataFrame containing product data.
    df_order (DataFrame): DataFrame containing order data.
    df_customer (DataFrame): DataFrame containing customer data.
    order_id_col (str): Column name for order ID in the order items and orders DataFrames.
    product_id_col (str): Column name for product ID in the order items and products DataFrames.
    product_name_col (str): Column name for product name in the products DataFrame.
    customer_id_col (str): Column name for customer ID in the customers and orders DataFrames.
    order_date_col (str): Column name for order date in the orders DataFrame.
    birth_date_col (str): Column name for birth date in the customers DataFrame.
    top_n (int): The number of top products to retrieve for each age group. Default is 5.

    Returns:
    None: Displays a plot of the top N most frequently ordered products for each age group.
    """
    try:
        logger.info("Starting the process to calculate the most frequently ordered products by customer age group.")

        # Check if required columns are present
        if not check_columns(df_order_items, [product_id_col, order_id_col]) or \
           not check_columns(df_product, [product_id_col, product_name_col]) or \
           not check_columns(df_order, [order_id_col, customer_id_col, order_date_col]) or \
           not check_columns(df_customer, [customer_id_col, birth_date_col]):
            logger.error("Required columns are missing in one of the DataFrames.")
            return

        # Step 1: Calculate customer age and create age groups
        current_year = 2024  # Update with the current year
        logger.info("Calculating customer age and creating age groups.")
        df_customer = df_customer.withColumn("age", current_year - year(col(birth_date_col)))

        df_customer = df_customer.withColumn(
            "age_group",
            when(col("age") < 20, "<20")
            .when((col("age") >= 20) & (col("age") < 30), "20-29")
            .when((col("age") >= 30) & (col("age") < 40), "30-39")
            .when((col("age") >= 40) & (col("age") < 50), "40-49")
            .otherwise("50+")
        )

        # Step 2: Join the tables: customers, orders, order items, and products
        logger.info("Joining customers, orders, order items, and products DataFrames.")
        df_combined = df_customer.join(df_order, customer_id_col) \
                                 .join(df_order_items, order_id_col) \
                                 .join(df_product, product_id_col)

        # Step 3: Group by age group and product, then count the number of orders
        logger.info("Aggregating order counts by age group and product.")
        df_age_group_product_count = df_combined.groupBy("age_group", product_name_col) \
                                                .agg(count(product_id_col).alias("order_count"))

        # Step 4: For each age group, find the top N products
        logger.info(f"Finding the top {top_n} products for each age group.")
        window_spec = Window.partitionBy("age_group").orderBy(col("order_count").desc())
        df_top_products_by_age_group = df_age_group_product_count.withColumn("rank", rank().over(window_spec)) \
                                                                 .filter(col("rank") <= top_n)

        # Step 5: Collect the data for plotting
        logger.info("Collecting the top products data for each age group.")
        top_products_by_age_group = df_top_products_by_age_group.collect()

        # Step 6: Prepare data for plotting
        logger.info("Preparing data for plotting.")
        age_groups = list(set([row['age_group'] for row in top_products_by_age_group]))
        age_groups.sort()  # Sort the age groups in ascending order

        fig, ax = plt.subplots(len(age_groups), 1, figsize=(10, 12))
        fig.tight_layout(pad=4.0)

        for idx, age_group in enumerate(age_groups):
            # Get top products for this age group
            group_data = [row for row in top_products_by_age_group if row['age_group'] == age_group]
            product_names = [row[product_name_col] for row in group_data]
            order_counts = [row['order_count'] for row in group_data]

            # Plot for each age group
            ax[idx].barh(product_names, order_counts, color='lightgreen')
            ax[idx].set_title(f"Top {top_n} Products for Age Group {age_group}")
            ax[idx].set_xlabel("Order Count")
            ax[idx].set_ylabel("Product Name")

        plt.show()

        logger.info("Successfully plotted the top products by age group.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

#Average Product catergory Revenue
def get_average_revenue_by_category(df_order_items: DataFrame, df_product: DataFrame, 
                                     product_id_col: str, quantity_col: str, price_col: str, 
                                     category_col: str) -> DataFrame:
    """
    Calculates the average revenue of all products by category and returns a DataFrame.

    Parameters:
    df_order_items (DataFrame): DataFrame containing order items data.
    df_product (DataFrame): DataFrame containing product data with categories.
    product_id_col (str): Column name for product ID.
    quantity_col (str): Column name for quantity in order items.
    price_col (str): Column name for price in order items.
    category_col (str): Column name for product category.

    Returns:
    DataFrame: A DataFrame with average revenue by category.
    """
    try:
        logger.info("Starting the process to calculate average revenue by product category.")

        # Check if required columns are present
        if not check_columns(df_order_items, [product_id_col, quantity_col, price_col]) or \
           not check_columns(df_product, [product_id_col, category_col]):
            logger.error("Required columns are missing in one of the DataFrames.")
            return None

        # Step 1: Rename the 'price' column in order_items to avoid ambiguity
        logger.info("Renaming price column in order items to avoid ambiguity.")
        df_order_items = df_order_items.withColumnRenamed(price_col, "order_price")

        # Step 2: Join the order_items DataFrame with the products DataFrame on 'product_id'
        logger.info("Joining order items with products to calculate revenue.")
        df_combined = df_order_items.join(df_product, df_order_items[product_id_col] == df_product[product_id_col], how='inner')

        # Step 3: Calculate revenue for each product in each order
        logger.info("Calculating revenue for each product (order_price * quantity).")
        df_combined = df_combined.withColumn("revenue", col("order_price") * col(quantity_col))

        # Step 4: Group by category and calculate the average revenue
        logger.info("Grouping by product category and calculating average revenue.")
        df_category_avg_revenue = df_combined.groupBy(category_col).agg(
            floor(avg("revenue")).alias("average_revenue")
        )

        # Step 5: Return the DataFrame with average revenue by category
        logger.info("Successfully calculated average revenue by category. Returning the DataFrame.")
        return df_category_avg_revenue

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None


#Plot the top n average revenue products by category
def plot_top_n_average_revenue(df: DataFrame, category_col: str, revenue_col: str, top_n: int = 10) -> None:
    """
    Plots the top N product categories by average revenue.

    Parameters:
    df (DataFrame): DataFrame containing category and revenue data.
    category_col (str): Column name for the product category.
    revenue_col (str): Column name for the average revenue.
    top_n (int): Number of top categories to plot. Default is 10.

    Returns:
    None
    """
    try:
        logger.info(f"Starting the plotting process for top {top_n} categories by average revenue.")

        # Check if required columns are present
        if not check_columns(df, [category_col, revenue_col]):
            logger.error("Required columns are missing in the DataFrame.")
            return

        # Sort the DataFrame by average revenue in descending order and select the top N
        logger.info(f"Sorting the DataFrame by {revenue_col} and selecting the top {top_n}.")
        top_n_df = df.orderBy(col(revenue_col).desc()).limit(top_n)

        # Collect the data for plotting
        logger.info("Collecting the data for plotting.")
        top_n_data = top_n_df.collect()

        # Extract category names and average revenues for plotting
        categories = [row[category_col] for row in top_n_data]
        average_revenues = [row[revenue_col] for row in top_n_data]

        # Plot the top N categories by average revenue
        logger.info("Plotting the top categories by average revenue.")
        plt.figure(figsize=(10, 6))
        plt.bar(categories, average_revenues, color='skyblue')

        plt.title(f"Top {top_n} Product Categories by Average Revenue")
        plt.xlabel("Category")
        plt.ylabel("Average Revenue")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

        logger.info("Successfully completed plotting.")

    except Exception as e:
        logger.error(f"An error occurred while plotting: {e}")
        raise


#Revenue Product wise
def get_total_revenue_by_product(df_order_items: DataFrame, df_products: DataFrame, 
                                 product_id_col: str, quantity_col: str, price_col: str, 
                                 product_name_col: str) -> DataFrame:
    """
    Calculates the total revenue for each product by multiplying price and quantity.

    Parameters:
    df_order_items (DataFrame): DataFrame containing order items, including product_id, price, and quantity.
    df_products (DataFrame): DataFrame containing product details, including product_id and product_name.
    product_id_col (str): Column name for product ID.
    quantity_col (str): Column name for quantity in order items.
    price_col (str): Column name for price in products or order items.
    product_name_col (str): Column name for product name.

    Returns:
    DataFrame: A DataFrame with product_id, product_name, and total revenue columns.
    """
    try:
        logger.info("Starting the process to calculate total revenue by product.")

        # Check if required columns are present in both DataFrames
        if not check_columns(df_order_items, [product_id_col, quantity_col]) or \
           not check_columns(df_products, [product_id_col, product_name_col, price_col]):
            logger.error("Required columns are missing in one of the DataFrames.")
            return None

        # Alias the DataFrames to avoid ambiguity
        df_order_items = df_order_items.alias("order_items")
        df_products = df_products.alias("products")

        # Join the order_items DataFrame with products to get product names
        logger.info("Joining order_items with products to get product names.")
        df_combined = df_order_items.join(df_products, 
                                          df_order_items[product_id_col] == df_products[product_id_col], 
                                          how='inner')

        # Calculate the revenue for each product (price * quantity)
        logger.info("Calculating revenue for each product.")
        df_combined = df_combined.withColumn("revenue", col(f"products.{price_col}") * col(f"order_items.{quantity_col}"))

        # Group by product_id and product_name, summing the revenue
        logger.info("Grouping by product_id and product_name to calculate total revenue.")
        df_total_revenue = df_combined.groupBy(f"products.{product_id_col}", f"products.{product_name_col}").agg(
            _sum("revenue").alias("total_revenue")
        )

        # Log the success message
        logger.info("Successfully calculated total revenue by product.")

        return df_total_revenue

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


#Inventory Status
def get_inventory_status(df_order_items: DataFrame, df_products: DataFrame, 
                         product_id_col: str, product_name_col: str, quantity_col: str, 
                         price_col: str, inventory_col: str) -> DataFrame:
    """
    Determines the inventory status for each product based on revenue and inventory conditions.

    - If revenue > 25,000 and inventory > 100, set status to "To be ordered."
    - If inventory < 50, set status to "To be ordered."
    - If inventory = 0, set status to "Out of stock."
    - For inventory > 100, set status to "In stock."

    Parameters:
    df_order_items (DataFrame): DataFrame containing order items, including product_id, price, and quantity.
    df_products (DataFrame): DataFrame containing product details, including product_id, product_name, price, and inventory.
    product_id_col (str): Column name for product ID.
    product_name_col (str): Column name for product name.
    quantity_col (str): Column name for quantity in order items.
    price_col (str): Column name for price in products or order items.
    inventory_col (str): Column name for inventory in products.

    Returns:
    DataFrame: A DataFrame with product_id, product_name, inventory, total_revenue, and inventory_status.
    """
    try:
        logger.info("Starting the process to determine inventory status by product.")

        # Check if required columns are present in both DataFrames
        if not check_columns(df_order_items, [product_id_col, quantity_col, price_col]) or \
           not check_columns(df_products, [product_id_col, product_name_col, inventory_col]):
            logger.error("Required columns are missing in one of the DataFrames.")
            return None

        # Call the get_total_revenue_by_product function to calculate revenue
        logger.info("Calling get_total_revenue_by_product to get revenue data.")
        df_total_revenue = get_total_revenue_by_product(df_order_items, df_products, 
                                                        product_id_col, quantity_col, 
                                                        price_col, product_name_col)

        # Alias the DataFrames to avoid ambiguity
        df_total_revenue = df_total_revenue.alias("total_revenue")
        df_products = df_products.alias("products")

        # Join the total revenue DataFrame with the products DataFrame to get inventory data
        logger.info("Joining total revenue with products to get inventory and determine status.")
        df_combined = df_total_revenue.join(
            df_products,
            df_total_revenue[f"total_revenue.{product_id_col}"] == df_products[f"products.{product_id_col}"],
            how="inner"
        )

        # Determine the inventory status based on revenue and inventory thresholds
        logger.info("Assigning inventory status based on revenue and inventory conditions.")
        df_combined = df_combined.withColumn(
            "inventory_status",
            when((col("total_revenue.total_revenue") > 25000) & (col(f"products.{inventory_col}") > 100), "To be ordered")
            .when(col(f"products.{inventory_col}") < 50, "To be ordered")
            .when(col(f"products.{inventory_col}") == 0, "Out of stock")
            .otherwise("In stock")
        )

        # Select relevant columns for the final DataFrame
        df_inventory_status = df_combined.select(
            col(f"products.{product_id_col}").alias("product_id"),
            col(f"products.{product_name_col}").alias("product_name"),
            col(f"products.{inventory_col}").alias("inventory"),
            col("total_revenue.total_revenue"),
            col("inventory_status")
        )

        logger.info("Successfully determined inventory status for each product.")
        return df_inventory_status

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

#Plot Inventory Status
def plot_inventory_status_percentage(df_order_items: DataFrame, df_products: DataFrame, 
                                     product_id_col: str, product_name_col: str, 
                                     quantity_col: str, price_col: str, inventory_col: str):
    """
    Calls get_inventory_status to retrieve product-wise inventory status and 
    plots the percentage of products for each inventory status, including handling cases
    where some statuses may have 0% representation.

    Parameters:
    df_order_items (DataFrame): DataFrame containing order items data (product_id, quantity, price).
    df_products (DataFrame): DataFrame containing product data (product_id, product_name, inventory).
    product_id_col (str): Column name for product ID.
    product_name_col (str): Column name for product name.
    quantity_col (str): Column name for quantity in order items.
    price_col (str): Column name for price in products or order items.
    inventory_col (str): Column name for inventory in products.

    Returns:
    None
    """
    try:
        logger.info("Starting the process to get inventory status and plot it.")

        # Check if the required columns are present in both DataFrames
        if not check_columns(df_order_items, [product_id_col, quantity_col, price_col]) or \
           not check_columns(df_products, [product_id_col, product_name_col, inventory_col]):
            logger.error("Required columns are missing in one of the DataFrames.")
            return None

        # Call get_inventory_status to get the DataFrame with inventory status
        logger.info("Calling get_inventory_status to retrieve inventory status for each product.")
        df_inventory_status = get_inventory_status(df_order_items, df_products, 
                                                   product_id_col, product_name_col, 
                                                   quantity_col, price_col, inventory_col)

        # Group by inventory_status and count the number of products
        logger.info("Grouping by inventory status and counting products.")
        df_status_count = df_inventory_status.groupBy("inventory_status").count()

        # Collect the data for plotting
        logger.info("Collecting the data for plotting.")
        status_data = df_status_count.collect()

        # Ensure all possible statuses are represented
        all_statuses = ["In stock", "To be ordered", "Out of stock"]
        status_dict = {status: 0 for status in all_statuses}
        
        for row in status_data:
            status_dict[row['inventory_status']] = row['count']

        statuses = list(status_dict.keys())
        counts = list(status_dict.values())

        # Calculate the total count for percentage calculation
        total_products = sum(counts)
        if total_products == 0:
            percentages = [0] * len(counts)
        else:
            percentages = [(count / total_products) * 100 for count in counts]

        # Plot the percentage of products for each inventory status
        logger.info("Plotting the inventory status percentages.")
        plt.figure(figsize=(10, 6))
        bars = plt.bar(statuses, percentages, color=['skyblue', 'lightgreen', 'salmon'])

        # Add percentage labels on top of each bar
        for bar, percentage in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{percentage:.2f}%', ha='center', va='bottom', fontsize=12, color='black')

        # Improve the appearance of the plot
        plt.title("Percentage of Products by Inventory Status", fontsize=16)
        plt.xlabel("Inventory Status", fontsize=14)
        plt.ylabel("Percentage of Products (%)", fontsize=14)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

        logger.info("Successfully completed plotting.")

    except Exception as e:
        logger.error(f"An error occurred while plotting: {e}")
        raise

# Customer Life Time Value
def calculate_customer_lifetime_value(df_order_items: DataFrame, df_orders: DataFrame, df_customers: DataFrame, 
                                      order_id_col: str, customer_id_col: str, price_col: str, quantity_col: str, 
                                      first_name_col: str, last_name_col: str) -> DataFrame:
    """
    Calculates the Customer Lifetime Value (CLV) for each customer by summing the total revenue generated by each customer,
    and includes customer names in the result.

    Parameters:
    df_order_items (DataFrame): DataFrame containing order items with product_id, price, and quantity.
    df_orders (DataFrame): DataFrame containing orders with order_id and customer_id.
    df_customers (DataFrame): DataFrame containing customer details with customer_id, first_name, and last_name.
    order_id_col (str): Column name for order ID.
    customer_id_col (str): Column name for customer ID.
    price_col (str): Column name for price in the order items.
    quantity_col (str): Column name for quantity in the order items.
    first_name_col (str): Column name for customer's first name.
    last_name_col (str): Column name for customer's last name.

    Returns:
    DataFrame: A DataFrame with customer_id, customer_name, and CLV.
    """
    try:
        logger.info("Starting the process to calculate CLV for each customer.")

        # Check if required columns are present in all DataFrames
        if not check_columns(df_order_items, [order_id_col, price_col, quantity_col]) or \
           not check_columns(df_orders, [order_id_col, customer_id_col]) or \
           not check_columns(df_customers, [customer_id_col, first_name_col, last_name_col]):
            logger.error("Required columns are missing in one of the DataFrames.")
            return None

        # Alias the DataFrames to avoid ambiguity
        df_order_items = df_order_items.alias("order_items")
        df_orders = df_orders.alias("orders")
        df_customers = df_customers.alias("customers")

        # Join the order_items DataFrame with orders to get customer_id
        logger.info("Joining order_items with orders to get customer_id.")
        df_combined = df_order_items.join(df_orders, df_order_items[order_id_col] == df_orders[order_id_col], how='inner')

        # Calculate revenue for each order item (price * quantity)
        logger.info("Calculating revenue for each order item.")
        df_combined = df_combined.withColumn("revenue", col(f"order_items.{price_col}") * col(f"order_items.{quantity_col}"))

        # Group by customer_id and sum the revenue to get CLV
        logger.info("Grouping by customer_id and calculating total revenue (CLV).")
        df_clv = df_combined.groupBy(f"orders.{customer_id_col}").agg(
            _sum("revenue").alias("CLV")
        )

        # Create a full name column in the customers DataFrame
        df_customers = df_customers.withColumn("customer_name", concat_ws(" ", col(first_name_col), col(last_name_col)))

        # Join with df_customers to get customer names
        logger.info("Joining CLV DataFrame with customers to get customer names.")
        df_result = df_clv.join(df_customers, df_clv[customer_id_col] == df_customers[customer_id_col], how='inner') \
                          .select(df_customers[customer_id_col], df_customers["customer_name"], df_clv["CLV"])

        # Log the success message
        logger.info("Successfully calculated CLV and joined with customer names.")

        return df_result

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

# Plot top n premium customer based on clv
def plot_top_n_premium_customers(
    df_customer_clv: DataFrame, 
    customer_name_col: str, 
    clv_col: str, 
    top_n: int = 10
) -> None:
    """
    Plots the top N premium customers based on Customer Lifetime Value (CLV).

    Parameters:
    df_customer_clv (DataFrame): DataFrame containing customer details and CLV.
    customer_name_col (str): Column name for customer names.
    clv_col (str): Column name for CLV values.
    top_n (int): Number of top customers to plot. Default is 10.

    Returns:
    None
    """
    try:
        logger.info(f"Starting the plotting process for top {top_n} premium customers based on CLV.")

        # Check if required columns are present
        required_columns = [customer_name_col, clv_col]
        if not check_columns(df_customer_clv, required_columns):
            logger.error("Required columns are missing in the DataFrame.")
            return

        # Get the top N premium customers
        logger.info(f"Sorting the DataFrame by {clv_col} and selecting the top {top_n} customers.")
        top_customers_df = df_customer_clv.orderBy(col(clv_col).desc()).limit(top_n)

        # Collect the data for plotting
        logger.info("Collecting the data for plotting.")
        top_customers = top_customers_df.collect()

        # Extract customer names and CLVs for plotting
        customer_names = [row[customer_name_col] for row in top_customers]
        clvs = [row[clv_col] for row in top_customers]

        # Plot the top N premium customers by CLV
        logger.info("Plotting the top premium customers by CLV.")
        plt.figure(figsize=(12, 8))
        bars = plt.bar(customer_names, clvs, color='teal', edgecolor='black')

        # Add labels to the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2, height,
                f'{height:,.2f}', ha='center', va='bottom',
                fontsize=10, color='black'
            )

        plt.title(f"Top {top_n} Premium Customers by CLV")
        plt.xlabel("Customer")
        plt.ylabel("CLV")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        logger.info("Successfully completed plotting.")

    except Exception as e:
        logger.error(f"An error occurred while plotting: {e}")
        raise

#Yearly Revenue
def calculate_yearly_revenue(
    df_orders: DataFrame, 
    df_order_items: DataFrame, 
    order_id_col: str, 
    order_date_col: str, 
    price_col: str, 
    quantity_col: str, 
    revenue_col: str = 'revenue', 
    year_col: str = 'year', 
    total_revenue_col: str = 'total_revenue'
) -> DataFrame:
    """
    Calculates the total revenue for each year.

    Parameters:
    df_orders (DataFrame): DataFrame containing orders.
    df_order_items (DataFrame): DataFrame containing order items.
    order_id_col (str): Column name for order ID.
    order_date_col (str): Column name for order date.
    price_col (str): Column name for price in order items.
    quantity_col (str): Column name for quantity in order items.
    revenue_col (str): Column name for calculated revenue.
    year_col (str): Column name for the year.
    total_revenue_col (str): Column name for total revenue.

    Returns:
    DataFrame: A DataFrame with year and total revenue columns.
    """
    try:
        logger.info("Starting the calculation of yearly revenue.")

        # Check if required columns are present in both DataFrames
        if not check_columns(df_orders, [order_id_col, order_date_col]) or \
           not check_columns(df_order_items, [order_id_col, price_col, quantity_col]):
            logger.error("Required columns are missing in one or more of the DataFrames.")
            return

        # Calculate revenue for each order item
        logger.info("Calculating revenue for each order item.")
        df_order_items = df_order_items.withColumn(revenue_col, col(price_col) * col(quantity_col))

        # Join the order_items DataFrame with orders to get order_date
        logger.info("Joining order items with orders to get the order_date.")
        df_combined = df_order_items.join(df_orders, on=order_id_col)

        # Extract year from order_date
        logger.info("Extracting the year from the order_date.")
        df_combined = df_combined.withColumn(year_col, year(col(order_date_col)))

        # Aggregate revenue by year
        logger.info("Aggregating total revenue by year.")
        df_yearly_revenue = df_combined.groupBy(year_col) \
                                       .agg(_sum(revenue_col).alias(total_revenue_col))

        logger.info("Successfully calculated yearly revenue.")
        return df_yearly_revenue

    except Exception as e:
        logger.error(f"An error occurred while calculating yearly revenue: {e}")
        raise

#plot for yearly revenue
def plot_yearly_revenue(
    df_yearly_revenue: DataFrame, 
    year_col: str, 
    total_revenue_col: str
) -> None:
    """
    Plots the total revenue for each year.

    Parameters:
    df_yearly_revenue (DataFrame): DataFrame containing yearly revenue data.
    year_col (str): Column name for the year in the DataFrame.
    total_revenue_col (str): Column name for total revenue in the DataFrame.

    Returns:
    None
    """
    try:
        logger.info("Starting the plotting process for yearly revenue.")

        # Check if required columns are present in the DataFrame
        if not check_columns(df_yearly_revenue, [year_col, total_revenue_col]):
            logger.error("Required columns are missing in the DataFrame.")
            return

        # Collect the data for plotting
        yearly_revenue_data = df_yearly_revenue.orderBy(year_col).collect()

        # Extract years and revenues for plotting
        years = [row[year_col] for row in yearly_revenue_data]
        revenues = [row[total_revenue_col] for row in yearly_revenue_data]

        # Plot the yearly revenue
        plt.figure(figsize=(12, 8))
        plt.bar(years, revenues, color='cornflowerblue', edgecolor='black')

        # Add labels to the bars
        for i, revenue in enumerate(revenues):
            plt.text(years[i], revenue, f'{revenue:,.2f}', ha='center', va='bottom', fontsize=10, color='black')

        plt.title("Yearly Revenue")
        plt.xlabel("Year")
        plt.ylabel("Total Revenue")
        plt.xticks(years)
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        logger.info("Successfully completed plotting yearly revenue.")

    except Exception as e:
        logger.error(f"An error occurred while plotting yearly revenue: {e}")
        raise

#Max Orders Per Year
def get_max_orders_per_year(
    df_orders: DataFrame, 
    order_date_col: str, 
    order_id_col: str, 
    year_col: str = "year", 
    num_orders_col: str = "num_orders", 
    max_orders_col: str = "max_orders"
) -> DataFrame:
    """
    Calculates the maximum number of orders for each year.

    Parameters:
    df_orders (DataFrame): DataFrame containing orders.
    order_date_col (str): Column name for the order date in the DataFrame.
    order_id_col (str): Column name for the order ID in the DataFrame.
    year_col (str): Column name for the extracted year. Default is 'year'.
    num_orders_col (str): Column name for the number of orders. Default is 'num_orders'.
    max_orders_col (str): Column name for the maximum number of orders. Default is 'max_orders'.

    Returns:
    DataFrame: A DataFrame with year and maximum number of orders per year.
    """
    try:
        logger.info("Starting the process to calculate maximum orders per year.")

        # Extract year from the order_date
        logger.info(f"Extracting year from the {order_date_col} column.")
        df_orders = df_orders.withColumn(year_col, year(col(order_date_col)))

        # Count the number of orders per year
        logger.info("Counting the number of orders per year.")
        df_orders_per_year = df_orders.groupBy(year_col) \
                                      .agg(count(order_id_col).alias(num_orders_col))

        # Find the maximum number of orders per year
        logger.info("Finding the maximum number of orders per year.")
        df_max_orders_per_year = df_orders_per_year.groupBy(year_col) \
                                                   .agg({num_orders_col: "max"}) \
                                                   .withColumnRenamed(f"max({num_orders_col})", max_orders_col)

        # Log the success message
        logger.info("Successfully calculated maximum orders per year.")

        return df_max_orders_per_year

    except Exception as e:
        logger.error(f"An error occurred while calculating maximum orders per year: {e}")
        raise

#Plot Max Orders
def plot_max_orders_per_year(
    df_max_orders: DataFrame, 
    year_col: str, 
    max_orders_col: str
) -> None:
    """
    Plots the maximum number of orders per year.

    Parameters:
    df_max_orders (DataFrame): DataFrame containing year and maximum number of orders per year.
    year_col (str): Column name for the year in the DataFrame.
    max_orders_col (str): Column name for the maximum number of orders in the DataFrame.

    Returns:
    None
    """
    try:
        logger.info("Starting the plotting process for maximum orders per year.")

        # Collect the data for plotting
        logger.info("Collecting the data for plotting.")
        data = df_max_orders.orderBy(year_col).collect()

        # Extract years and maximum orders for plotting
        years = [row[year_col] for row in data]
        max_orders = [row[max_orders_col] for row in data]

        # Plot the maximum orders per year
        logger.info("Plotting the maximum orders per year.")
        plt.figure(figsize=(12, 8))
        bars = plt.bar(years, max_orders, color='royalblue', edgecolor='black')

        # Add labels to the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2, height,
                f'{height:,.0f}', ha='center', va='bottom',
                fontsize=10, color='black'
            )

        plt.title("Maximum Orders Per Year")
        plt.xlabel("Year")
        plt.ylabel("Maximum Number of Orders")
        plt.xticks(years, rotation=45, ha="right")
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        logger.info("Successfully completed plotting.")

    except Exception as e:
        logger.error(f"An error occurred while plotting: {e}")
        raise

#Average Ratings per products
def categorize_reviews(
    df_reviews: DataFrame, 
    df_products: DataFrame, 
    review_id_col: str, 
    rating_col: str, 
    product_id_col: str, 
    product_name_col: str
) -> DataFrame:
    """
    Categorizes products based on their average review ratings and includes product names.

    Parameters:
    df_reviews (DataFrame): DataFrame containing reviews, including product_id and rating.
    df_products (DataFrame): DataFrame containing product details, including product_id and product_name.
    review_id_col (str): Column name for the review ID in df_reviews.
    rating_col (str): Column name for the rating in df_reviews.
    product_id_col (str): Column name for the product ID in both df_reviews and df_products.
    product_name_col (str): Column name for the product name in df_products.

    Returns:
    DataFrame: A DataFrame with product_id, product_name, average_rating, and rating_category.
    """
    try:
        # Calculate the average rating for each product
        df_avg_ratings = df_reviews.groupBy(product_id_col) \
                                  .agg(floor(avg(rating_col)).alias("average_rating"))

        # Join with product details to get product names
        df_avg_ratings_with_names = df_avg_ratings.join(df_products, on=product_id_col, how="inner") \
                                                  .select(product_id_col, product_name_col, "average_rating")

        # Categorize products based on their average rating
        categorize_expr = when(col("average_rating") == 1, 'poor') \
            .when(col("average_rating") == 2, 'average') \
            .when(col("average_rating") == 3, 'satisfactory') \
            .when(col("average_rating") == 4, 'highly-satisfied') \
            .when(col("average_rating") == 5, 'excellent') \
            .otherwise('unknown')

        df_categorized_reviews = df_avg_ratings_with_names.withColumn("rating_category", categorize_expr)

        return df_categorized_reviews

    except Exception as e:
        raise

#Plot Top n product ratings
def plot_top_n_product_ratings(df_categorize_reviews: DataFrame, product_id_col: str, product_name_col: str, average_rating_col: str, rating_category_col: str, top_n: int = 10) -> None:
    """
    Plots the top N products with the highest average ratings.

    Parameters:
    df (DataFrame): DataFrame containing product_id, average_rating, rating_category, and product_name.
    product_id_col (str): Column name for the product ID.
    product_name_col (str): Column name for the product name.
    average_rating_col (str): Column name for the average rating.
    rating_category_col (str): Column name for the rating category.
    top_n (int): Number of top products to plot based on average rating. Default is 10.

    Returns:
    None
    """
    try:
        logger.info(f"Starting the plotting process for top {top_n} products with highest average ratings.")

        # Check if required columns are present
        if not check_columns(df_categorize_reviews, [product_id_col, product_name_col, average_rating_col, rating_category_col]):
            logger.error("Required columns are missing in the DataFrame.")
            return

        # Sort the DataFrame by average rating in descending order and select the top N
        logger.info(f"Sorting the DataFrame by {average_rating_col} and selecting the top {top_n}.")
        top_n_df = df_categorize_reviews.orderBy(col(average_rating_col).desc()).limit(top_n)

        # Collect the data for plotting
        logger.info("Collecting data from DataFrame.")
        data = top_n_df.collect()

        # Extract data for plotting
        product_names = [row[product_name_col] for row in data]
        average_ratings = [row[average_rating_col] for row in data]
        categories = [row[rating_category_col] for row in data]

        logger.info("Extracted product names, ratings, and categories.")

        # Plot the top N products by average rating
        plt.figure(figsize=(14, 8))
        bars = plt.barh(product_names, average_ratings, color='mediumseagreen', edgecolor='black')

        # Add labels to the bars
        for bar, category in zip(bars, categories):
            width = bar.get_width()
            plt.text(
                min(width + 0.05, 5.5),  # Adjust the position to keep the text within the plot
                bar.get_y() + bar.get_height() / 2,
                f'{int(width)}\n{category}', ha='left', va='center',
                fontsize=10, color='black'
            )

        plt.xlim(0, 5.5)  # Set a maximum x limit for ratings to avoid text overflow
        plt.title(f"Top {top_n} Products with Highest Average Ratings")
        plt.xlabel("Average Rating")
        plt.ylabel("Product Name")
        plt.tight_layout()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        logger.info(f"Plotting top {top_n} products with highest average ratings.")
        plt.show()
        logger.info("Plot displayed successfully.")

    except Exception as e:
        logger.error(f"An error occurred while plotting: {e}")
        raise
