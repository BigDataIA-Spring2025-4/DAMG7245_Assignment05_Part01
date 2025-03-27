import os
import numpy as np
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
import yfinance as yf
from snowflake.connector.pandas_tools import write_pandas

load_dotenv()

# Load Snowflake credentials
required_env_vars = {
    'SNOWFLAKE_ACCOUNT': os.getenv('SNOWFLAKE_ACCOUNT'),
    'SNOWFLAKE_USER': os.getenv('SNOWFLAKE_USER'),
    'SNOWFLAKE_PASSWORD': os.getenv('SNOWFLAKE_PASSWORD'),
    'SNOWFLAKE_ROLE': os.getenv('SNOWFLAKE_ROLE'),
    'SNOWFLAKE_DB': os.getenv('SNOWFLAKE_DB'),
    'SNOWFLAKE_WAREHOUSE': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'SNOWFLAKE_SCHEMA': os.getenv('SNOWFLAKE_SCHEMA')
}

missing_vars = [key for key, value in required_env_vars.items() if value is None]
if missing_vars:
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}. Please set them before running the script.")

# Connect to Snowflake
try:
    conn = snowflake.connector.connect(
        account=required_env_vars['SNOWFLAKE_ACCOUNT'],
        user=required_env_vars['SNOWFLAKE_USER'],
        password=required_env_vars['SNOWFLAKE_PASSWORD'],
        role=required_env_vars['SNOWFLAKE_ROLE'],
        warehouse=required_env_vars['SNOWFLAKE_WAREHOUSE'],
        database=required_env_vars['SNOWFLAKE_DB'],
        schema=required_env_vars['SNOWFLAKE_SCHEMA']
    )
    cur = conn.cursor()
    cur.execute("SELECT current_version()")
    print(f"✅ Successfully connected to Snowflake. Version: {cur.fetchone()[0]}")
except snowflake.connector.errors.Error as e:
    print(f"❌ Snowflake connection failed: {str(e)}")
    exit(1)

###########################################################################
# Fetch data for Nvidia
companies = {'Nvidia': 'NVDA'}
start_date = '2020-01-01'
data_list = []

for company_name, ticker in companies.items():
    stock_data = yf.download(ticker, start=start_date)
    info = yf.Ticker(ticker).info

    shares_outstanding = info.get("sharesOutstanding", np.nan)
    trailing_eps = info.get("trailingEps", np.nan)
    dividend_rate = info.get("dividendRate", np.nan)
    total_debt = info.get("totalDebt", np.nan)
    free_cashflow = info.get("freeCashflow", np.nan)
    total_revenue = info.get("totalRevenue", np.nan)

    # Build company data
    company_data = {
        'Company': company_name,
        'Ticker': ticker,
        'Date': stock_data.index,
        'Open': stock_data['Open'].values,
        'High': stock_data['High'].values,
        'Low': stock_data['Low'].values,
        'Close': stock_data['Close'].values,
        'Volume': stock_data['Volume'].values,
        'marketCap': np.where(np.isnan(shares_outstanding), np.nan, stock_data['Close'].values * shares_outstanding),
        'EPS': trailing_eps,
        'PERatio': np.where(np.isnan(trailing_eps), np.nan, stock_data['Close'].values / trailing_eps),
        'Beta': info.get('beta', np.nan),
        'dividendYield': np.where((np.isnan(dividend_rate) | np.isnan(shares_outstanding)), np.nan, (dividend_rate * shares_outstanding) / (stock_data['Close'].values * shares_outstanding)),
        'enterpriseValue': np.where((np.isnan(total_debt) | np.isnan(free_cashflow)), np.nan, (stock_data['Close'].values * shares_outstanding) + total_debt - free_cashflow),
        'priceToSalesTrailing12Months': np.where(np.isnan(total_revenue), np.nan, (stock_data['Close'].values * shares_outstanding) / total_revenue)
    }

    # Convert to DataFrame
    company_df = pd.DataFrame(company_data)
    data_list.append(company_df)

# Concatenate all data
data = pd.concat(data_list, ignore_index=True)
data.columns = data.columns.str.upper()
data['DATE'] = pd.to_datetime(data['DATE']).dt.date
data = data.reset_index(drop=True)

# Handle None values (Snowflake needs NaN)
data = data.astype(object).where(pd.notnull(data), None)

# Create table in Snowflake
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {required_env_vars['SNOWFLAKE_SCHEMA']}.NVIDIA_VALUATION_V2 (
    COMPANY VARCHAR(255),
    TICKER VARCHAR(10),
    DATE DATE,
    OPEN FLOAT,
    HIGH FLOAT,
    LOW FLOAT,
    CLOSE FLOAT,
    VOLUME FLOAT,
    MARKETCAP FLOAT,
    EPS FLOAT,
    PERATIO FLOAT,
    BETA FLOAT,
    DIVIDENDYIELD FLOAT,
    ENTERPRISEVALUE FLOAT,
    PRICETOSALESTRAILING12MONTHS FLOAT
);
"""
conn.cursor().execute(create_table_sql)

# Insert data into Snowflake
success, nchunks, nrows, _ = write_pandas(
    conn=conn,
    df=data,
    table_name='NVIDIA_VALUATION_V2',
    database=required_env_vars['SNOWFLAKE_DB'],
    schema=required_env_vars['SNOWFLAKE_SCHEMA'],
    overwrite=False,
    auto_create_table=False
)

print(f"✅ Inserted {nrows} rows across {nchunks} chunks.")

# Close connection
cur.close()
conn.close()
print("🔻 Connection closed.")
