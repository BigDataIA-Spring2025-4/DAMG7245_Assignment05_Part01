import re
import pandas as pd
import yfinance as yf
import snowflake.connector
from dotenv import load_dotenv
from snowflake.connector.pandas_tools import write_pandas
import os
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

def snowflake_pandaspush(create_table_ddl, data, table_name, reset_index_flag=True):
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
    except Exception as e:
        print(f"❌ Snowflake connection failed: {str(e)}")
        return

    try:
        cur.execute(create_table_ddl)
        if reset_index_flag:
            data = data.reset_index().rename(columns={'index': 'DATE'})
        data.columns = [col.upper() for col in data.columns]
        if 'TICKER' not in data.columns:
            data['TICKER'] = 'NVDA'
        data = data.rename(columns=lambda x: x.strip().replace(" ", ""))
        data.columns = [col.upper().replace(' ', '_') for col in data.columns]
        if table_name.find("HISTORICAL") != -1:
            data = data[['TICKER', 'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 
                         'VOLUME', 'DIVIDENDS', 'STOCKSPLITS']]
        success, nchunks, nrows, _ = write_pandas(
            conn=conn,
            df=data,
            table_name=table_name,
            database=required_env_vars['SNOWFLAKE_DB'],
            schema=required_env_vars['SNOWFLAKE_SCHEMA'],
            overwrite=True,
            auto_create_table=False,
            quote_identifiers=False,
            use_logical_type=True
        )
        print(f"✅ Inserted {nrows} rows across {nchunks} chunks into table {table_name}.")
        
    except Exception as e:
        print(f"❌ Error during Snowflake operation: {str(e)}")
    finally:
        cur.close()
        conn.close()
        print("🔻 Connection closed.")

def historical_data(ticker, period):
    period_clean = period.replace("y", "Y").replace(" ", "_")
    table_name = f"{ticker}_HISTORICAL_{period_clean}"
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        TICKER VARCHAR(10),
        DATE TIMESTAMP_NTZ,
        OPEN FLOAT,
        HIGH FLOAT,
        LOW FLOAT,
        CLOSE FLOAT,
        VOLUME FLOAT,
        DIVIDENDS FLOAT,
        STOCKSPLITS FLOAT
    );
    """
    data = yf.Ticker(ticker).history(period=period)
    snowflake_pandaspush(create_table_sql, data, table_name, reset_index_flag=True)
    return data

def balance_sheet_data(ticker):
    bs = yf.Ticker(ticker).balance_sheet.T
    bs.index.name = "DATE"
    bs = bs.reset_index()
    bs['TICKER'] = ticker
    bs.columns = [col.upper().strip().replace(" ", "_") for col in bs.columns]
    bs = bs.loc[:, ~bs.columns.duplicated()]
    table_name = f"{ticker}_BALANCE_SHEET"
    ddl_columns = []
    for col in bs.columns:
        if col == "TICKER":
            ddl_columns.append(f"{col} VARCHAR(10)")
        elif col == "DATE":
            ddl_columns.append(f"{col} TIMESTAMP_NTZ")
        else:
            ddl_columns.append(f"{col} FLOAT")
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {', '.join(ddl_columns)}
    );
    """
    snowflake_pandaspush(create_table_sql, bs, table_name, reset_index_flag=False)
    return bs

def clean_column_name(col_name):
    """Sanitize column names for SQL compatibility."""
    col_name = col_name.upper().strip().replace(" ", "_")  
    col_name = re.sub(r"[^A-Z0-9_]", "", col_name)  
    return col_name

def financials_data(ticker):
    data = yf.Ticker(ticker)
    fin = data.financials

    fin = fin.T  
    fin.reset_index(inplace=True)
    fin.rename(columns={'index': 'DATE'}, inplace=True)

    fin['TICKER'] = ticker  
    fin.columns = [clean_column_name(col) for col in fin.columns]

    table_name = f"{ticker}_FINANCIALS"
    ddl_columns = []
    for col in fin.columns:
        if col == "TICKER":
            ddl_columns.append(f'"{col}" VARCHAR(100)')
        elif col == "DATE":
            ddl_columns.append(f"{col} TIMESTAMP_NTZ")
        else:
            ddl_columns.append(f'"{col}" FLOAT')
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {', '.join(ddl_columns)}
    );
    """
    snowflake_pandaspush(create_table_sql, fin, table_name, reset_index_flag=False)
    return fin

def main():
    ticker = 'NVDA'
    period = "5y"
    hist_data = historical_data(ticker, period)
    print(f"\nLoaded historical data from {period}:")
    print(hist_data.head())
    bs_data = balance_sheet_data(ticker)
    print(f"\nLoaded balance sheet data for {ticker}:")
    print(bs_data.head())
    print("\nColumns in balance sheet data:")
    fs_data = financials_data(ticker)
    print(fs_data) 

if __name__ == "__main__":
    main()




