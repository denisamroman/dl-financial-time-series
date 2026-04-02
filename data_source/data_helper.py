# Import the SimFin library
import simfin as sf

# Import the names used for easy access to SimFin's data-columns
from simfin.names import *

# Import the Yahoo! Finance library
# yfinance==0.2.1 was required for sourcing the fundamental data
import yfinance as yf

# Import data manipulation library
import pandas as pd

import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import random 

def getdata(filename, frequency):
        df = pd.read_csv('./data_source/data/'+frequency+'/'+filename+'.csv')
        df.Date = pd.to_datetime(df.Date)
        df = df.set_index('Date', drop=True)
        return df

# API key to access simfin
# Add your simfin API key manually before you continue
def get_simfin_api_key():
    # Get simfin API keys
    api_key= "cqvorQ2ttQ4NHb1tBou1vTgSRF7CUvXQ"
    if api_key=="Copy your simfin API key here":
        return print("Your API key is missing.")
    else:
        return api_key

# This function returns Fred API key
# Add your FRED API key manually before you continue
def get_fred_api():
    # Get Fred API
    api_key= "959b9fea3199e9debc45e80d4061f216"
    if api_key=="Copy your fred API key here":
        return print("Your API key is missing.")
    else:
        return api_key  
    
def get_stock_data(asset_ticker, start_date, end_date):
    stock_data = yf.download(asset_ticker, start=start_date, end=end_date, auto_adjust=True)
    return stock_data

# Define the utility to get the fundamental data for any asset ticker
def get_fundamental_data(asset_ticker):    
    # Define the asset ticker for yfinance
    asset_yf_ticker = yf.Ticker(asset_ticker)
    
    # SimFin data-directory
    sf.set_data_dir('./data_source/data/simfin_data')

    # SimFin API key
    simfin_api_key = get_simfin_api_key()
    sf.config.set_api_key(api_key=simfin_api_key)

    # Set the market as US
    market = 'us'

    # Fetch quarterly income statements for all the tickers in SimFin database
    income_data_simfin_all_stocks = sf.load_income(variant='quarterly', market=market)

    # Get the quarterly income statement for the ticker
    income_data_simfin = income_data_simfin_all_stocks.loc[asset_ticker,:]

    # Get the quarterly income statement from yfinance
    income_data_yfinance = asset_yf_ticker.quarterly_income_stmt.T

    """
    The mapping dictionary stores the column names from yfinance as key and
    the corresponding column name from Simfin as the value
    """
    income_data_mapping_dict =\
        {
            "Total Revenue" : "Revenue",
            "Operating Income" : "Operating Income (Loss)",
            "Pretax Income" : "Pretax Income (Loss)",
            "Net Income" : "Net Income"
        }

    # Rename the columns in the income_data_yfinance DataFrame
    income_data_yfinance = \
        income_data_yfinance.rename(columns=income_data_mapping_dict)

    # The final column list for the merged DataFrame
    income_data_column_heads = [i[1] for i in income_data_mapping_dict.items()]

    # Trim the income_data_simfin to the final column list
    income_data_simfin = income_data_simfin[income_data_column_heads]

    # Trim the income_data_yfinance to the final column list
    income_data_yfinance = income_data_yfinance[income_data_column_heads]

    # Sort the income_data_yfinance to match the order as in the SimFin DataFrame
    income_data_yfinance = income_data_yfinance.sort_index(ascending=True)

    # Join the two DataFrames
    income_data = income_data_simfin.append(income_data_yfinance)

    # Fetch quarterly balance sheets for all the tickers in SimFin database
    balance_sheet_simfin_all_stocks = sf.load_balance(variant='quarterly', market=market)

    # Get the quarterly balance sheets for the ticker
    balance_sheet_simfin = balance_sheet_simfin_all_stocks.loc[asset_ticker,:]

    # Get the quarterly balance sheets from yfinance
    balance_sheet_yfinance = asset_yf_ticker.quarterly_balance_sheet.T

    """
    The mapping dictionary stores the column names from yfinance as key and
    the corresponding column name from Simfin as the value
    """
    balance_sheet_mapping_dict =\
        {
            "Total Assets" : "Total Assets",
            "Total Liabilities Net Minority Interest" : "Total Liabilities",
            "Current Assets" : "Total Current Assets",
            "Current Liabilities" : "Total Current Liabilities",
            "Stockholders Equity" : "Total Equity",
            "Retained Earnings" : "Retained Earnings",
            "Long Term Debt" : "Long Term Debt"
        }

    # Rename the columns in the balance_sheet_yfinance DataFrame
    balance_sheet_yfinance = \
        balance_sheet_yfinance.rename(columns=balance_sheet_mapping_dict)

    # The final column list for the merged DataFrame
    balance_sheet_column_heads = [i[1] for i in balance_sheet_mapping_dict.items()]

    # Trim the balance_sheet_simfin to the final column list
    balance_sheet_simfin = balance_sheet_simfin[balance_sheet_column_heads]

    # Trim the balance_sheet_yfinance to the final column list
    balance_sheet_yfinance = balance_sheet_yfinance[balance_sheet_column_heads]

    # Sort the balance_sheet_yfinance to match the order as in the SimFin DataFrame
    balance_sheet_yfinance = balance_sheet_yfinance.sort_index(ascending=True)

    # Join the two DataFrames
    balance_sheet_data = balance_sheet_simfin.append(balance_sheet_yfinance)

    # Fetch quarterly cash flow statements for all the tickers in SimFin database
    cashflow_simfin_all_stocks = sf.load_cashflow(variant='quarterly', market=market)

    # Get the quarterly cash flow statements for the ticker
    cashflow_simfin = cashflow_simfin_all_stocks.loc[asset_ticker,:]

    # Get the quarterly cash flow statements from yfinance
    cashflow_yfinance = asset_yf_ticker.quarterly_cashflow.T

    """
    The mapping dictionary stores the column names from yfinance as key and
    the corresponding column name from Simfin as the value
    """
    cashflow_mapping_dict =\
        {
            "Operating Cash Flow" : "Net Cash from Operating Activities",
            "Investing Cash Flow" : "Net Cash from Investing Activities",
            "Financing Cash Flow" : "Net Cash from Financing Activities"
        }

    # Rename the columns in the cashflow_yfinance DataFrame
    cashflow_yfinance = \
        cashflow_yfinance.rename(columns=cashflow_mapping_dict)

    # The final column list for the merged DataFrame
    cashflow_column_heads = [i[1] for i in cashflow_mapping_dict.items()]

    # Trim the cashflow_simfin to the final column list
    cashflow_simfin = cashflow_simfin[cashflow_column_heads]

    # Trim the cashflow_yfinance to the final column list
    cashflow_yfinance = cashflow_yfinance[cashflow_column_heads]

    # Sort the balance_sheet_yfinance to match the order as in the SimFin DataFrame
    cashflow_yfinance = cashflow_yfinance.sort_index(ascending=True)

    # Join the two DataFrames
    cashflow_data = cashflow_simfin.append(cashflow_yfinance)

    # Return the income statement, balance sheet and cashflow data
    return income_data, balance_sheet_data, cashflow_data

# create function to group trade hours
def daypart(hour):
    if hour in [9,10,11]:
        return "morning"
    elif hour in [12,13]:
        return "noon"
    elif hour in [14,15,16,17,18,19]:
        return "afternoon"
    else:
        return "evening"

def get_trading_view_data(filename, frequency):
    df = pd.read_csv('./data_source/data/'+frequency+'/'+filename+'.csv')
    df.datetime = pd.to_datetime(df.datetime)
    df = (
        df.set_index('datetime', drop=True)
    )
    
    # add days
    df['days'] = df.index.day_name()

    # add dayparts
    df['hours'] = df.index.hour
    df['hours'] = df['hours'].apply(daypart)

    return df

