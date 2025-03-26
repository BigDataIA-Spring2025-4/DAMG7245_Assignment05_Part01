import yfinance as yf
import pandas as pd
import time

# List of companies with their ticker symbols
companies = {
    'Nvidia': 'NVDA'
}

#Define the start date
start_date = '2020-01-01'

# List to collect all data
data_list = []

# Fetch data for each company
for company_name, ticker in companies.items():
    # Get historical market data from start_date to today
    stock_data = yf.download(ticker, start=start_date)
    
    # Flatten the MultiIndex columns by combining the index levels
    stock_data.columns = [f'{col[0]}_{col[1]}' for col in stock_data.columns]

    # Check the flattened columns
    print(f"Columns for {company_name} ({ticker}):", stock_data.columns)

    # If 'Adj Close' is available, use it. Otherwise, fall back to 'Close'
    adj_close_column = 'Adj Close' if 'Adj Close' in stock_data.columns else 'Close'

    # Get company info
    info = yf.Ticker(ticker).info
    
    # Create the company data as a list of dictionaries
    company_data = {
        'Company': company_name,
        'Ticker': ticker,
        'Date': stock_data.index,
        'Open': stock_data['Open_' + ticker].values,
        'High': stock_data['High_' + ticker].values,
        'Low': stock_data['Low_' + ticker].values,
        'Close': stock_data['Close_' + ticker].values,
        adj_close_column: stock_data[adj_close_column + '_' + ticker].values,  # Use the adjusted close if available
        'Volume': stock_data['Volume_' + ticker].values,
        'Market Cap': info.get('marketCap', None),
        'PE Ratio': info.get('trailingPE', None),
        'Beta': info.get('beta', None),
        'EPS': info.get('trailingEps', None),
        'Forward PE': info.get('forwardPE', None),
        'Revenue': info.get('revenue', None),
        'Gross Profit': info.get('grossProfits', None),
        'Operating Income': info.get('operatingIncome', None),
        'Net Income': info.get('netIncomeToCommon', None),
        'Debt to Equity': info.get('debtToEquity', None),
        'Return on Equity (ROE)': info.get('returnOnEquity', None),
        'Current Ratio': info.get('currentRatio', None),
        'Dividends Paid': info.get('lastDividendValue', None),
        'Dividend Yield': info.get('dividendYield', None),
        'Quarterly Revenue Growth': info.get('revenueGrowth', None),
        'Analyst Recommendation': info.get('recommendationKey', None),
        'Target Price': info.get('targetMeanPrice', None),
        'Free Cash Flow': info.get('freeCashflow', None),
        'Operating Margin': info.get('operatingMargins', None),
        'Profit Margin': info.get('profitMargins', None),
        'Cash Ratio': info.get('cashRatio', None),
        'Quick Ratio': info.get('quickRatio', None),
        'Price to Book Ratio': info.get('priceToBook', None),
        'Enterprise Value': info.get('enterpriseValue', None),
        'Total Debt': info.get('totalDebt', None),
        'Total Assets': info.get('totalAssets', None),
        'Total Equity': info.get('totalStockholderEquity', None),
        'Beta (5Y)': info.get('beta', None),  # Historical volatility over 5 years
        'Annual Dividend Rate': info.get('dividendRate', None),  # Annual dividends
        'Trailing Twelve Months (TTM) Revenue': info.get('revenueTTM', None),
        'Trailing Twelve Months (TTM) EBITDA': info.get('ebitdaTTM', None),
        'Trailing Twelve Months (TTM) Earnings': info.get('netIncomeTTM', None)
    }

    # Convert the company data to a DataFrame and append it to the list
    company_df = pd.DataFrame(company_data)

    # Add the DataFrame for the company to the list
    data_list.append(company_df)

    # Sleep for a short time to prevent hitting the Yahoo Finance API rate limit
    time.sleep(2)

# Combine all the data into one DataFrame
data = pd.concat(data_list, ignore_index=True)

# Display the dataset
print(data)

# Save to CSV
data.to_csv('Nvidia_v1.csv', index=False)
