import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
import yfinance as yf
from snowflake.connector.pandas_tools import write_pandas

load_dotenv()

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

    # Execute a test query
    cur.execute("SELECT current_version()")
    result = cur.fetchone()

    print(f"✅ Successfully connected to Snowflake.")
    print(f"🔹 Snowflake version: {result[0]}")

except snowflake.connector.errors.Error as e:
    print(f"❌ Snowflake connection failed: {str(e)}")
    exit(1)

###########################################################################
# Scraping for Nvidia
# List of companies with their ticker symbols
companies = {
    'Nvidia': 'NVDA'
}

start_date = '2020-01-01'
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
        'totalRevenue': info.get('totalRevenue', None),
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
    data.columns = data.columns.str.upper()
    data['DATE'] = pd.to_datetime(data['DATE']).dt.date
    data = data.reset_index(drop=True)

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {required_env_vars['SNOWFLAKE_SCHEMA']}.NVIDIA_VALUATION (
        COMPANY VARCHAR(255),
        TICKER VARCHAR(10),
        DATE DATE,
        OPEN FLOAT,
        HIGH FLOAT,
        LOW FLOAT,
        CLOSE FLOAT,
        VOLUME FLOAT,
        MARKETCAP FLOAT,
        PERATIO FLOAT,
        BETA FLOAT,
        EPS FLOAT,
        FORWARDPE FLOAT,
        PEGRATIO FLOAT,
        PRICETOSALESTRAILING12MONTHS FLOAT,
        ENTERPRISETOREVENUE FLOAT,
        ENTERPRISETOEBITDA FLOAT,
        TOTALREVENUE FLOAT,
        GROSSPROFITS FLOAT,
        OPERATINGINCOME FLOAT,
        NETINCOME FLOAT,
        DEBTTOEQUITY FLOAT,
        RETURNONEQUITY FLOAT,
        CURRENTRATIO FLOAT,
        LASTDIVIDENDVALUE FLOAT,
        DIVIDENDYIELD FLOAT,
        QUARTERLYREVENUEGROWTH FLOAT,
        ANALYSTRECOMMENDATION VARCHAR(50),
        TARGETMEANPRICE FLOAT,
        FREECASHFLOW FLOAT,
        OPERATINGMARGINS FLOAT,
        PROFITMARGINS FLOAT,
        CASHRATIO FLOAT,
        QUICKRATIO FLOAT,
        PRICETOBOOKRATIO FLOAT,
        ENTERPRISEVALUE FLOAT,
        TOTALDEBT FLOAT,
        TOTALASSETS FLOAT,
        TOTALSTOCKHOLDEREQUITY FLOAT,
        ANNUALDIVIDENDRATE FLOAT,
        REVENUETTM FLOAT,
        EBITDATTM FLOAT,
        NETINCOMETTM FLOAT,
        LOAD_TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
    );
    """

    conn.cursor().execute(create_table_sql)

    # Insert data into Snowflake
    success, nchunks, nrows, _ = write_pandas(
        conn=conn,
        df=data,
        table_name='NVIDIA_VALUATION',
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