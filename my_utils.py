import os
import csv
import pandas as pd
import logging
import hashlib
import json

#Setup logging
logging.basicConfig(level=logging.INFO)

#List of common encodings
ENCODINGS = ['utf-8','latin1', 'ISO-8859', 'cp1252', 'cp850', 'utf-16', 'utf-32']

def save_to_csv(file, output_dir, name, announce = True):
    ''' 
    Saves a dataframe to a CSV file.
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file.to_csv(os.path.join(output_dir, name))
    if announce: 
            logging.info(f'Successfully created file {name} in {output_dir}')
            print(f'{"<" * 10} Processed data saved to {os.path.join(output_dir, name)} {">" * 10}')

def detect_delimiter(filename):
    '''
    Determines the delimiter used in a CSV file.
    '''
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    logging.info(f'Delimiter detected: {repr(delimiter)}')

def numeric_only(df):
    """
    Create a copy of the DataFrame with only numeric columns.
    """
    # Select numeric columns
    numeric_columns = df.select_dtypes(include='number')
    
    # Create a copy with only numeric columns
    df_numeric = numeric_columns.copy()
    
    return df_numeric

def object_df(df):
    """
    Create a copy of the DataFrame with only non-numeric aka object datatype columns.
    """
    # Select numeric columns
    obj_col = df.select_dtypes(include = 'object')
    
    # Create a copy with only numeric columns
    df_object = obj_col.copy()
    
    return df_object

def one_hot_encode_top_values(df, num_top_values=5):
    """
    Performs one-hot encoding using the top 'num_top_values' most common data points for each attribute.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    num_top_values (int): Number of top values to consider for each attribute.

    Returns:
    pd.DataFrame: New DataFrame with one-hot encoded columns.
    """
    df_copy = df.copy()

    encoded_columns = []

    for col in df.columns:
        top_values = df_copy[col].value_counts().nlargest(num_top_values).index
        for value in top_values:
            new_col_name = f"{col}_{value}"
            df_copy[new_col_name] = (df_copy[col] == value).astype(int)
            encoded_columns.append(new_col_name)

    return df_copy[encoded_columns]

    # parameters: original df, dummy df, and target_Attribute variable
def return_target_col(df, df_dummy, target_attribute):
    df_full = df_dummy
    df_full[target_attribute] = df[target_attribute]
    return df_full

def str_attribute_to_len(df, col_name):
    column_name = str(col_name)
    # Compute string lengths for the specified column
    string_lengths = df[column_name].astype(str).apply(len)

    return string_lengths