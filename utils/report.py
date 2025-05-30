from pyspark.sql.functions import *
from fpdf import FPDF
from utils.logging_setup import setup_logging

logger = setup_logging("/spark-data/logs/logger.log")
report_folder = "/spark-data/Report"

def analyze_dataframes(dataframes: dict, spark):
    """Function to analyze multiple DataFrames and return a consolidated report."""
    
    # List to store analysis results
    analysis_results = []
    
    # Iterate through the dictionary of DataFrames
    for df_name, df in dataframes.items():
        try:
            logger.info(f"Analyzing DataFrame: {df_name}")
            
            # Get row and column counts
            row_count = df.count()
            column_count = len(df.columns)
            
            # Calculate duplicate count
            duplicate_count = df.count() - df.dropDuplicates().count()

            # Iterate through each column of the DataFrame
            for column in df.columns:
                try:
                    # Calculate null counts (handling actual null, NaN, and the string 'null')
                    null_count = df.select(count(when(col(column).isNull() | (col(column) == 'null'), column))).collect()[0][0]
                    
                    # Calculate negative counts (only for numeric columns)
                    negative_count = 0
                    if df.schema[column].dataType.simpleString() in ('int', 'long', 'float', 'double'):
                        negative_count = df.filter(col(column) < 0).count()
                    
                    # Get the correct data type of the column
                    data_type = df.schema[column].dataType.simpleString()
                    
                    # Append the results as a tuple to the list
                    analysis_results.append((df_name, column, null_count, negative_count, data_type, row_count, column_count, duplicate_count))
                    
                except Exception as e:
                    logger.error(f"Error analyzing column {column} in DataFrame {df_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error analyzing DataFrame {df_name}: {e}")
    
    # Create a DataFrame from the results list
    try:
        schema = ['DataFrame Name', 'Column Name', 'Null Count', 'Negative Count', 'Data Type', 'Row Count', 'Column Count', 'Duplicate Count']
        result_df = spark.createDataFrame(analysis_results, schema=schema)
        logger.info("Analysis DataFrame created successfully.")
    except Exception as e:
        logger.error(f"Error creating analysis DataFrame: {e}")
        raise
    
    return result_df

def compare_dataframes_and_generate_report(before_df: DataFrame, after_df: DataFrame, report_path: str):
    """
    Compares two DataFrames' analyses and generates a report with changes.

    Parameters:
    before_df (DataFrame): Analysis DataFrame before cleaning.
    after_df (DataFrame): Analysis DataFrame after cleaning.
    report_path (str): Path to save the PDF report.
    """
    try:
        # Collect data from both DataFrames
        before_data = before_df.collect()
        after_data = after_df.collect()
        
        # Create dictionaries for quick lookups
        before_dict = {(row['DataFrame Name'], row['Column Name']): row for row in before_data}
        after_dict = {(row['DataFrame Name'], row['Column Name']): row for row in after_data}
        
        # Initialize PDF
        pdf = FPDF()
        
        # First Page: Data Cleaning Report
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Add title
        pdf.cell(200, 10, txt="Data Cleaning Report", ln=True, align='C')
        pdf.ln(10)
        
        # Add headers
        pdf.cell(40, 10, 'DataFrame Name', 1)
        pdf.cell(40, 10, 'Column Name', 1)
        pdf.cell(50, 10, 'Metrics', 1)
        pdf.cell(30, 10, 'Before', 1)
        pdf.cell(30, 10, 'After', 1)
        pdf.ln()
        
        # Compare before and after data
        all_keys = set(before_dict.keys()).union(after_dict.keys())
        
        for key in all_keys:
            df_name = key[0]
            column_name = key[1]
            
            before_row = before_dict.get(key, {})
            after_row = after_dict.get(key, {})

            try:
                # Null counts
                before_null_count = before_row['Null Count'] if 'Null Count' in before_row else 0
                after_null_count = after_row['Null Count'] if 'Null Count' in after_row else 0
                if before_null_count != after_null_count:
                    pdf.cell(40, 10, df_name, 1)
                    pdf.cell(40, 10, column_name, 1)
                    pdf.cell(50, 10, 'Null Count', 1)
                    pdf.cell(30, 10, f"{before_null_count}", 1)
                    pdf.cell(30, 10, f"{after_null_count}", 1)
                    pdf.ln()
                
                # Negative counts
                before_negative_count = before_row['Negative Count'] if 'Negative Count' in before_row else 0
                after_negative_count = after_row['Negative Count'] if 'Negative Count' in after_row else 0
                
                if (before_negative_count is not None and before_negative_count != 0) or after_negative_count != 0:
                    if before_negative_count != after_negative_count:
                        pdf.cell(40, 10, df_name, 1)
                        pdf.cell(40, 10, column_name, 1)
                        pdf.cell(50, 10, 'Negative Count', 1)
                        pdf.cell(30, 10, f"{before_negative_count if before_negative_count is not None else 0}", 1)
                        pdf.cell(30, 10, f"{after_negative_count}", 1)
                        pdf.ln()
                
                # Data types
                before_data_type = before_row['Data Type'] if 'Data Type' in before_row else ''
                after_data_type = after_row['Data Type'] if 'Data Type' in after_row else ''
                if before_data_type != after_data_type:
                    pdf.cell(40, 10, df_name, 1)
                    pdf.cell(40, 10, column_name, 1)
                    pdf.cell(50, 10, 'Data Type Change', 1)
                    pdf.cell(30, 10, f"{before_data_type}", 1)
                    pdf.cell(30, 10, f"{after_data_type}", 1)
                    pdf.ln()

            except Exception as e:
                logger.error(f"Error processing {df_name} - {column_name}: {e}")

        # New Page: DataFrame Row and Column Counts
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Add title for the new page
        pdf.cell(200, 10, txt="DataFrame Row and Column Counts", ln=True, align='C')
        pdf.ln(10)
        
        # Add headers
        pdf.cell(30, 10, 'DataFrame', 1)
        pdf.cell(25, 10, 'Row Before', 1)
        pdf.cell(25, 10, 'Row After', 1)
        pdf.cell(25, 10, 'Column Bef', 1)
        pdf.cell(25, 10, 'Column Af', 1)
        pdf.cell(30, 10, 'Duplicate Bef', 1)
        pdf.cell(25, 10, 'Duplicate Af', 1)
        pdf.ln()
        
        # Collect unique DataFrame names
        df_names = set(row['DataFrame Name'] for row in before_data)
        
        for df_name in df_names:
            before_row_count = before_df.filter(col("DataFrame Name") == df_name).agg({"Row Count": "max"}).collect()[0][0]
            after_row_count = after_df.filter(col("DataFrame Name") == df_name).agg({"Row Count": "max"}).collect()[0][0]
            before_column_count = before_df.filter(col("DataFrame Name") == df_name).agg({"Column Count": "max"}).collect()[0][0]
            after_column_count = after_df.filter(col("DataFrame Name") == df_name).agg({"Column Count": "max"}).collect()[0][0]
            before_duplicate_count = before_df.filter(col("DataFrame Name") == df_name).agg({"Duplicate Count": "max"}).collect()[0][0]
            after_duplicate_count = after_df.filter(col("DataFrame Name") == df_name).agg({"Duplicate Count": "max"}).collect()[0][0]
            
            pdf.cell(30, 10, df_name, 1)
            pdf.cell(25, 10, str(before_row_count), 1)
            pdf.cell(25, 10, str(after_row_count), 1)
            pdf.cell(25, 10, str(before_column_count), 1)
            pdf.cell(25, 10, str(after_column_count), 1)
            pdf.cell(30, 10, str(before_duplicate_count), 1)
            pdf.cell(25, 10, str(after_duplicate_count), 1)
            pdf.ln()

        # Save the PDF report
        pdf.output(report_path)
        logger.info(f"Report generated successfully at {report_path}.")
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
