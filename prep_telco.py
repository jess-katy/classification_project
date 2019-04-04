import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# def open_file(csv_file):
#     path = './'
#     df = pd.read_csv(path + {''})
#     return df

def peekatdata(df):
    print("\nRows & Columns:\n")
    print(df.shape)
    print("\nColumn Info:\n")
    print(df.info())
    print("\nFirst 5 rows:\n")
    print(df.head())
    print("\nLast 5 rows:\n")
    print(df.tail())
    print("\nMissing Values:\n")
    missing_vals = df.columns[df.isnull().any()]
    print(df.isnull().sum())
    print("\nSummary Stats:\n")
    print(df.describe())
    return df

def df_value_counts(df):
    for col in df.columns: 
        n = df[col].unique().shape[0] 
        col_bins = min(n,10) 
        if df[col].dtype in ['int64','float64'] and n > 10:
            print('%s:' % col)
            print(df[col].value_counts(bins=col_bins, sort=False)) 
        else: 
            print(df[col].value_counts()) 
        print('\n')
    return df

def percent_missing(df):
    missing_table = df.isnull().sum()/df.shape[0]*100
    return df

def charge_emtpy(df):
    df['total_charges'] = df['total_charges'].convert_objects(convert_numeric=True)
    df.total_charges.dropna(0, inplace=True)
    return df

def make_binary(df):
    df['churn'] == 'Yes'
    (df['churn'] == 'Yes').astype(int)
    df['churn'] = (df['churn'] == 'Yes').astype(int)
    return df

def tenure_yr(df): 
    df = df.assign(tenure_year=df.tenure/12).round(2)
    return df

def create_phone_id(df):
    df = df.replace({'phone_service': {'Yes': 1, 'No': 0}})
    df = df.replace({'multiple_lines': {'Yes': 1, 'No': 0, 'No phone service': 0}})
    df['phone_id'] = df['phone_service'].astype(int) + df['multiple_lines'].astype(int)
    return df

def create_household(df):
    df = df.replace({'partner': {'Yes': 1, 'No': 0}})
    df = df.replace({'dependents': {'Yes': 2, 'No': 0}})
    df['household_type_id'] = df['dependents'].astype(int) + df['partner'].astype(int)
    return df

def create_streaming(df):
    df = df.replace({'streaming_tv': {'Yes': 2, 'No': 1, 'No internet service': 0}})
    df = df.replace({'streaming_movies': {'Yes': 3, 'No': 1, 'No internet service': 0}})
    df['streaming_services'] = df['streaming_tv'].astype(int) + df['streaming_movies'].astype(int)
    return df

def create_security(df):
    df = df.replace({'online_security': {'Yes': 2, 'No': 1, 'No internet service': 0}})
    df = df.replace({'online_backup': {'Yes': 3, 'No': 1, 'No internet service': 0}})
    df['online_security_backup'] = df['online_security'].astype(int) + df['online_backup'].astype(int)
    return df

def split(df):
    train_df, test_df = train_test_split(df, test_size = .30, random_state = 123, stratify = df[['churn']])
    return train_df

def encode_data(df):
    for col in df.drop(columns=(['customer_id', 'total_charges', 'monthly_charges'])):
        encoder = LabelEncoder()
        encoder.fit(df[col])
        new_col = col + '_encode'
        df[new_col] = encoder.transform(df[col])

def encode_test_train(df):
    train_df = encode_data(train_df)
    test_df = encode_data(test_df)
    return train_df

def scale(train_df):
    scaler = MinMaxScaler()
    scaler.fit(train_df[['monthly_charges', 'total_charges']])
    train_df[['monthly_charges', 'total_charges']] = scaler.transform(train_df[['monthly_charges', 'total_charges']])
    test_df[['monthly_charges', 'total_charges']] = scaler.transform(test_df[['monthly_charges', 'total_charges']])
    return train_df

def prep_telco_data(df):
    # df = peekatdata(df)
    df = df_value_counts(df)
    df = percent_missing(df)
    df = charge_emtpy(df)
    df = make_binary(df)
    df = tenure_yr(df)
    df = create_phone_id(df)
    df = create_household(df)
    df = create_streaming(df)
    df = create_security(df)
    train_df = split(df)
    train_df = encode_data(train_df)
    # train_df = scale(train_df)
    return train_df