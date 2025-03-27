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

def snowflake_pandaspull(query):
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
    try:

        df = pd.read_sql(query, conn)
        print(f"✅ Data fetched successfully from Snowflake.")
        print(df.head())
        return df 

    except Exception as e:
        print(f"❌ Failed to fetch data: {str(e)}")

    finally:
        conn.close()
        print("🔻 Connection closed.")