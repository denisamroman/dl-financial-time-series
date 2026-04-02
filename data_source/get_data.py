# Import the SimFin library
import simfin as sf

# Import the names used for easy access to SimFin's data-columns
from simfin.names import *

# Import the yfinance library
import yfinance as yf
#from yahoo_fin.stock_info import *

# Import time manipulation library
from datetime import timedelta

# Import data manipulation library
import pandas as pd

from functools import reduce

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Import libraries
import sys
sys.path.append("..")

from data_helper import *
from fredapi import Fred

"""
Select to generate daily or hourly dataset
Select whether data needs to be downloaded from the public APIs for the daily frequency- 
if is_data_downloaded is set to False, data will be sourced from the public APIs (yfinance==0.2.1 required)
if is_data_downloaded is set to True, saved csv files will be used
"""
data_frequency='hourly' 
is_data_downloaded = True

"""
Hourly data for AAPL extracted from tradingview
"""
if (data_frequency == 'hourly'):
    df = get_trading_view_data('/eth_1h_4y', data_frequency)
    df.to_csv('./data_source/data/' + data_frequency + '/data_1h.csv')

elif (data_frequency == 'daily'):
    """
    Fetch stock, fundamental, and market data for AAPL
    Clean Data and missing values
    """

    # Define the stock ticker
    stock_ticker_symbol = 'AAPL'
    start_date = '2015-03-31'
    end_date = '2022-12-31'

    if(is_data_downloaded == False):  
        """ Fetch the stock data """

        stock_data = get_stock_data(stock_ticker_symbol,start_date,end_date)
        stock_data.to_csv('./data_source/data/' + data_frequency + '/apple_adj.csv')

        """ Fetch the fundamental data """

        income_data,\
            balance_sheet_data,\
                cashflow_data = get_fundamental_data(stock_ticker_symbol)

        income_data.to_csv('./data_source/data/' + data_frequency + '/income_data.csv')
        balance_sheet_data.to_csv('./data_source/data/' + data_frequency + '/balance_sheet_data.csv')
        cashflow_data.to_csv('./data_source/data/' + data_frequency + '/cashflow_data.csv')

        """ Fetch the market data """

        fred = Fred(get_fred_api())

        # Get the US GDP
        # Series ID for US GDP: GDP
        gdp = fred.get_series('GDP')
        # Data is fetched in Billions USD, divide by 1000 to convert into Trillions
        gdp = gdp/1000
        gdp = pd.DataFrame({'Date':gdp.index, 'GDP':gdp.values})
        gdp = gdp[gdp.Date >= start_date]
        gdp['Date'] = gdp['Date'] - pd.to_timedelta(1, unit='d')
        gdp.to_csv('./data_source/data/' + data_frequency + '/gdp.csv', index = False)

        # Get the CPI
        # Series ID for CPI: CPIAUCSL
        cpi = fred.get_series('CPIAUCSL')
        cpi = pd.DataFrame({'Date':cpi.index, 'CPI':cpi.values})
        cpi = cpi[cpi.Date >= start_date]
        cpi['Date'] = cpi['Date'] - pd.to_timedelta(1, unit='d')
        cpi.to_csv('./data_source/data/' + data_frequency + '/cpi.csv', index = False)

        # Get the 3 month, 1 year and 10 year treasury rates
        # Series ID for Treasury Rates: DGS3MO, DGS1, DGS10
        treasury_3M = fred.get_series('DGS3MO')
        treasury_3M = pd.DataFrame({'Date':treasury_3M.index, 'Treasury_3M':treasury_3M.values})
        treasury_3M = treasury_3M[treasury_3M.Date >= start_date]
        treasury_3M = treasury_3M.set_index('Date', drop=True)

        treasury_1Y = fred.get_series('DGS1')
        treasury_1Y = pd.DataFrame({'Date':treasury_1Y.index, 'Treasury_1Y':treasury_1Y.values})
        treasury_1Y = treasury_1Y[treasury_1Y.Date >= start_date]
        treasury_1Y = treasury_1Y.set_index('Date', drop=True)

        treasury_10Y = fred.get_series('DGS10')
        treasury_10Y = pd.DataFrame({'Date':treasury_10Y.index, 'Treasury_10Y':treasury_10Y.values})
        treasury_10Y = treasury_10Y[treasury_10Y.Date >= start_date]
        treasury_10Y = treasury_10Y.set_index('Date', drop=True)

        # Combine 3m, 1y, and 10y rates into one table
        treasury = pd.concat([treasury_3M, treasury_1Y, treasury_10Y], axis = 1)

        # Bank holidays have missing data, fill the null values with the previous value
        treasury = treasury.fillna(method='ffill')
        treasury.to_csv('./data_source/data/' + data_frequency + '/treasury.csv')

        # Get the Unemployment Rate
        # Series ID for Unemployment Rate: UNRATE
        unemployment_rate = fred.get_series('UNRATE')
        unemployment_rate = pd.DataFrame({'Date':unemployment_rate.index, 'Unemployment Rate':unemployment_rate.values})
        unemployment_rate = unemployment_rate[unemployment_rate.Date >= start_date]
        unemployment_rate['Date'] = unemployment_rate['Date'] - pd.to_timedelta(1, unit='d')
        unemployment_rate.to_csv('./data_source/data/' + data_frequency + '/unemployment_rate.csv', index = False)

        eps_data = getdata('eps_data', data_frequency)

    else:
        # Stock Data
        stock_data = getdata('apple_adj', data_frequency)

        # Fundamental Data
        balance_sheet_data = getdata('balance_sheet_data', data_frequency)
        cashflow_data = getdata('cashflow_data', data_frequency)
        income_data = getdata('income_data', data_frequency)
        eps_data = getdata('eps_data', data_frequency)

        # Market Data
        cpi = getdata('cpi', data_frequency)
        gdp = getdata('gdp', data_frequency)
        treasury = getdata('treasury', data_frequency)
        unemployment_rate = getdata('unemployment_rate', data_frequency)

    dfs = [stock_data, balance_sheet_data, cashflow_data, income_data, eps_data, cpi, gdp, treasury, unemployment_rate]

    # Perform an outer join on the 'date' column
    merged_data = reduce(lambda left,right: pd.merge(left,right,on='Date',how='outer'), dfs)

    # Sort data in case there are na values in the stock data that was joined first
    sorted_data = merged_data.sort_values(by='Date')

    # As the frequnecy of the fundamental and most of the market data is not daily, perform linear interpolation to fill missing values
    # Alternative option - forwardfill the missing values
    # Alternative option was time interpolation, however given that the stock data is not available for some days, and weekends, this was avoided
    
    columns = sorted_data.columns.drop(stock_data.columns)
    sorted_data[columns] = sorted_data[columns].interpolate()

    # Clean the dataset and save it to csv
    data = sorted_data.dropna().drop_duplicates().reset_index()
    data = data.set_index('Date', drop=True)

    data.to_csv('./data_source/data/'+data_frequency+'/data_1d.csv')

else:
    print('Selected data frequency is not available.')