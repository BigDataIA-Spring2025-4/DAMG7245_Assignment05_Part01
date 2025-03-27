import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
import yfinance as yf
from snowflake.connector.pandas_tools import write_pandas

load_dotenv()

# List of companies with their ticker symbols
companies = {
    'Nvidia': 'NVDA'
}

#Define the start date
start_date = '2020-01-01'

# List to collect all data
data_list = []

for company_name, ticker in companies.items():
    
    stock_data = yf.download(ticker, start=start_date)
    stock_data.columns = [f'{col[0]}_{col[1]}' for col in stock_data.columns]
    print(f"Columns for {company_name} ({ticker}):", stock_data.columns)
    info = yf.Ticker(ticker).info
    
    company_data = {
        'Company': company_name,
        'Ticker': ticker,
        'Date': stock_data.index,
        'Open': stock_data['Open_' + ticker].values,
        'High': stock_data['High_' + ticker].values,
        'Low': stock_data['Low_' + ticker].values,
        'Close': stock_data['Close_' + ticker].values,
        'Volume': stock_data['Volume_' + ticker].values,
        'marketCap': info.get('marketCap', None),
        'PERatio': info.get('trailingPE', None),
        'Beta': info.get('beta', None),
        'EPS': info.get('trailingEps', None),
        'forwardPE': info.get('forwardPE', None),
        "pegRatio": info.get("pegRatio", None),
        "priceToSalesTrailing12Months": info.get("priceToSalesTrailing12Months", None),
        "enterpriseToRevenue": info.get("enterpriseToRevenue", None),
        "enterpriseToEbitda": info.get("enterpriseToEbitda", None),
        'Revenue': info.get('revenue', None),
        'grossProfits': info.get('grossProfits', None),
        'operatingIncome': info.get('operatingIncome', None),
        'netIncome': info.get('netIncomeToCommon', None),
        'debtToEquity': info.get('debtToEquity', None),
        'returnOnEquity': info.get('returnOnEquity', None),
        'currentRatio': info.get('currentRatio', None),
        'lastDividendValue': info.get('lastDividendValue', None),
        'dividendYield': info.get('dividendYield', None),
        'QuarterlyrevenueGrowth': info.get('revenueGrowth', None),
        'AnalystRecommendation': info.get('recommendationKey', None),
        'targetMeanPrice': info.get('targetMeanPrice', None),
        'freeCashflow': info.get('freeCashflow', None),
        'operatingMargins': info.get('operatingMargins', None),
        'profitMargins': info.get('profitMargins', None),
        'cashRatio': info.get('cashRatio', None),
        'quickRatio': info.get('quickRatio', None),
        'priceToBookRatio': info.get('priceToBook', None),
        'enterpriseValue': info.get('enterpriseValue', None),
        'TotalDebt': info.get('totalDebt', None),
        'TotalAssets': info.get('totalAssets', None),
        'totalStockholderEquity': info.get('totalStockholderEquity', None),
        'AnnualDividendRate': info.get('dividendRate', None),
        'revenueTTM': info.get('revenueTTM', None),
        'ebitdaTTM': info.get('ebitdaTTM', None),
        'netIncomeTTM': info.get('netIncomeTTM', None)
    }

    company_df = pd.DataFrame(company_data)
    data_list.append(company_df)

data = pd.concat(data_list, ignore_index=True)

print(data)

data.to_csv('Nvidia_v1.csv', index=False)
