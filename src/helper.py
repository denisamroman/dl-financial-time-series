import pandas as pd
import numpy as np
import random
import datetime as dt
import ta
import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin

# define seed
def set_seeds(seed=42): 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Function to create features (predictors) list based on standard deviation and returns
def get_features(df):
    for r in range(10, 30, 5):
        df['Ret_'+str(r)] = df.close.rolling(r).sum()
        df['Std_'+str(r)] = df.close.rolling(r).std()
    # Drop NaN values
    df.dropna(inplace=True)
    return df

# Function to extract technical features using pandas-ta
def get_features_ta(df, columns_base):
    columns = df.columns
    columns_to_eliminate = columns[~columns.isin(columns_base)]
    stock_data = df.copy()
    stock_data.drop(columns_to_eliminate, axis=1, inplace=True)
    
    stock_data.ta.strategy('All')
    stock_data.drop(['HILOl_13_21', 'HILOs_13_21', 'PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'PSARaf_0.02_0.2', 'QQEl_14_5_4.236', 'QQEs_14_5_4.236', 'SUPERTl_7_3.0', 'SUPERTs_7_3.0'], axis=1, inplace=True)
    stock_data.drop(columns_base, axis=1, inplace=True)
    
    data = pd.concat([df, stock_data], axis = 1)

    data = data[200:]
    # backfill columns to address missing values
    data = data.bfill(axis=1)
    print(data.isna().sum())

    return data

# Function to extract financial ratios from the financial dataset
def get_features_fa(df):
    # Calculate the current ratio
    df['Current Ratio'] = df['Total Current Assets']/ df['Total Current Liabilities']
    # Calculate the return on equity
    df['Return On Equity'] = (df['Net Income'] / df['Total Equity']) * 100
    # Calculate the profit margin
    df['Profit Margin'] = (df['Net Income'] / df['Revenue']) * 100
    # Calculate the debt to equity ratio
    df['DE Ratio'] = df['Long Term Debt'] / df['Total Equity']
    # Calculate the price to earnings ratio
    df['PE Ratio'] = df['Close'] / df['Actual EPS']
    # Calculate the difference between actual vs. the estimated Earnings per Share
    df['Est/Act EPS'] = df['Estimated EPS'] - df['Actual EPS']
    # Drop the fundamental data used to determine the ratios
    """
    df.drop([
        'Total Assets','Total Liabilities','Total Current Assets','Total Current Liabilities','Total Equity','Retained Earnings','Long Term Debt','Net Cash from Operating Activities',
        'Net Cash from Investing Activities','Net Cash from Financing Activities','Revenue','Operating Income (Loss)','Pretax Income (Loss)','Net Income','Estimated EPS','Actual EPS'], axis=1, inplace=True)
    """
    print(df.isna().sum())
    return df

# Function to create the target label
def get_target(df, freq):
    if(freq=='daily'):
        x = 1 # change it for classing balance to 1.0009
        df['Target'] = np.where(df['Close'].shift(-1) > x * df['Open'].shift(-1), 1, 0) 
        df = df[:-1] # to take care of 1day ahead prediction
    else:
        x = 1 # change it for classing balance to 0.99991
        df['Target'] = np.where(df['close'].shift(-1) > x * df['open'].shift(-1), 1, 0) 
        df = df[:-1] # to take care of 1day ahead prediction
    return df

# Analyse percentile for returns to help set the threshold for the negatives
def get_pos_threshold(df):
    a = np.array(df['Close'].pct_change())
    a = a[~np.isnan(a)]
    b = a[:(len(a)*4//5)]
    c = a[len(a)//5:]
    print(np.percentile(a,50))
    print(np.percentile(b,50))
    print(np.percentile(c,50))
    neg, pos = np.bincount(df['Target'])
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

def generate_sequence(X, y, sequence_length):
        
        # create X & y data array
        X_new = []
        y_new = []

        for i in range(sequence_length, len(X), sequence_length):
            X_new.append(X[i - sequence_length:i])
            y_new.append(y[i])
        
        # Converting x_train and y_train to Numpy arrays
        return np.array(X_new), np.array(y_new)

def get_dataset_split(data, testsize, seqlen, use_time_series_generator, scaler, numfeat):
    X = data.drop(['Target'], axis = 1)
    y = data['Target']

    # Splitting the datasets into training and testing data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, shuffle=False)

    # scale the training dataset
    X_train = scaler.fit_transform(X_train)
    # scale the test dataset
    X_test = scaler.transform(X_test)

    if (use_time_series_generator):
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # Output the train and test data size
        print(f"Train and Test Size {X_train.shape}, {y_train.shape}, {X_test.shape}, {y_test.shape}")

    else:
        X_train, y_train = generate_sequence(X_train, y_train, seqlen)
        print(f'X_train: {X_train.shape}, y_train {y_train.shape}')
        X_test, y_test = generate_sequence(X_test, y_test, seqlen)
        print(f'X_test: {X_test.shape}, y_test {y_test.shape}')

        # reshaping array
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], numfeat))
        y_train = y_train[:, np.newaxis] 
        # check the array size
        print(f'X_train Shape: {X_train.shape}, y_train {y_train.shape}')

        # reshaping array
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], numfeat))
        y_test = y_test[:, np.newaxis] 
        # check the array size
        print(f'X_test Shape: {X_test.shape}, y_test {y_test.shape}')

    return X_train, y_train, X_test, y_test

# class weight function
def cwts(dfs):
    c0, c1 = np.bincount(dfs)
    w0=(1/c0)*(len(dfs))/2 
    w1=(1/c1)*(len(dfs))/2 
    return {0: w0, 1: w1}

# create custom day transformer 
class DayTransformer(BaseEstimator, TransformerMixin):
                                   
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        self.data = pd.DataFrame(
            {
        'WeekDay': ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            }
        )
        self.daysnum = np.array(self.data.index+1)
        return self
    
    
    def transform(self, X): # X is a dataframe
        Xt = X.copy()
        pi = np.pi
        num = Xt.index.weekday+1
        
        Xt['dsin'] = np.sin(2 * pi * num / np.max(self.daysnum))
        Xt['dcos'] = np.cos(2 * pi * num / np.max(self.daysnum))
        Xt = Xt.drop(['days'], axis=1)
        
        return Xt

    
# create custom time transformer 
class TimeTransformer(BaseEstimator, TransformerMixin):
                                   
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        self.data = pd.DataFrame(
            {
        'DayParts': ["afternoon","morning","noon"]
            }
        )
        self.timenum = np.array(self.data.index+1)
        return self
    
    
    def transform(self, X):
        Xt = X.copy()
        pi = np.pi
        num = Xt.hours.apply(lambda x: 1 if x=='afternoon' else (2 if x=='morning' else 3))
        Xt['tsin'] = np.sin(2 * pi * num / np.max(self.timenum))
        Xt['tcos'] = np.cos(2 * pi * num / np.max(self.timenum))
        Xt = Xt.drop(['hours'], axis=1)
        
        return Xt