import pandas as pd
import math
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler

#These functions were aided by ChatGPT
def norm_df(df):
    df_normalized = df.copy()

    # Normalize each column using Min-Max normalization
    for col in df_normalized.columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)

    return df_normalized

def numeric_only(df):
    """
    Create a copy of the DataFrame with only numeric columns.
    """
    # Select numeric columns
    numeric_columns = df.select_dtypes(include='number')
    
    # Create a copy with only numeric columns
    df_numeric = numeric_columns.copy()
    
    return df_numeric

def drop_duplicates(df): 
    print('\nDuplicate row removal:')
    print('Sample count before: ', len(df.index))
    df_no_duplicates = df.drop_duplicates()
    print('Sample count after: ', len(df_no_duplicates.index))

    return df_no_duplicates

def drop_null(df):
    df_no_nulls = df.dropna()
    print('\nNull row removal:')
    print('Sample count before: ', len(df.index))
    print('Sample count after: ', len(df_no_nulls.index))
    
    return df_no_nulls

def drop_out_of_domain(df):
    # Get only numeric data to identify rows with out-of-domain properties
    df_numeric = numeric_only(df)
    
    # Calculate the mean and standard deviation of all row means
    print('\nOut-of-domain row removal:')
    all_rows_mean = df_numeric.mean(axis=1)
    all_rows_mean_mean = all_rows_mean.mean()
    all_rows_mean_std = all_rows_mean.std()
    threshold_std = 2
    threshold = all_rows_mean_mean + threshold_std * all_rows_mean_std
    print('Threshold =', threshold_std, ' standard deviations')

    out_of_domain_indices = []
    
    # Iterate over rows and check for out-of-domain properties
    for idx, row in df_numeric.iterrows():
        row_mean = row.mean()
        if row_mean > threshold:
            out_of_domain_indices.append(idx)
            #print(f"Row {idx} has out-of-domain properties.") 
    
    df_reduced = df.drop(out_of_domain_indices)
    print('Number of rows with out-of-domain properties: ',len(out_of_domain_indices))
    print('\nSample count before: ', len(df.index))
    print('Sample count after: ', len(df_reduced.index))

    return df_reduced

# Get target attribute's column into Y
def set_y(df, target_attribute):
    Y = df.loc[:, target_attribute]
    return Y

# Get non-target attribute columns into X
def set_x(df, target_attribute):
    X = df.loc[:, df.columns != target_attribute]
    return X

def normalize(X):
    # Normalize data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X

def remove_non_cat(df, non_cat_col):
    
    filtered_columns = [col for col in df.columns if col not in non_cat_col]
    filtered_df = df[filtered_columns]

    return filtered_df
