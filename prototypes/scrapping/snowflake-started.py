import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
import yfinance as yf
from snowflake.connector.pandas_tools import write_pandas

# Load environment variables
load_dotenv()

# Required Environment Variables
required_env_vars = {
    'SNOWFLAKE_ACCOUNT': os.getenv('SNOWFLAKE_ACCOUNT'),
    'SNOWFLAKE_USER': os.getenv('SNOWFLAKE_USER'),
    'SNOWFLAKE_PASSWORD': os.getenv('SNOWFLAKE_PASSWORD'),
    'SNOWFLAKE_ROLE': os.getenv('SNOWFLAKE_ROLE'),
    'SNOWFLAKE_DB': os.getenv('SNOWFLAKE_DB'),
    'SNOWFLAKE_WAREHOUSE': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'SNOWFLAKE_SCHEMA': os.getenv('SNOWFLAKE_SCHEMA')
}

# Check for missing environment variables
missing_vars = [key for key, value in required_env_vars.items() if value is None]
if missing_vars:
    raise ValueError(f"❌ Missing environment variables: {', '.join(missing_vars)}. Please set them before running the script.")

# Try to establish the Snowflake connection
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

    # Create a cursor object
    cur = conn.cursor()

    # Execute a test query
    cur.execute("SELECT current_version()")
    result = cur.fetchone()

    print(f"✅ Successfully connected to Snowflake.")
    print(f"🔹 Snowflake version: {result[0]}")

except snowflake.connector.errors.Error as e:
    print(f"❌ Snowflake connection failed: {str(e)}")
    exit(1)  # Stop execution if connection fails

# Fetch NVIDIA data
ticker = yf.Ticker("NVDA")
data = ticker.info

# Ensure data is fetched
if not data:
    raise ValueError("❌ Failed to fetch data from Yahoo Finance. Check network or API status.")

# Extract valuation metrics
valuation_data = {
    "MARKET_CAP": data.get("marketCap", None),
    "ENTERPRISE_VALUE": data.get("enterpriseValue", None),
    "TRAILING_PE": data.get("trailingPE", None),
    "FORWARD_PE": data.get("forwardPE", None),
    "PEG_RATIO": data.get("pegRatio", None),
    "PRICE_SALES": data.get("priceToSalesTrailing12Months", None),
    "EV_REVENUE": data.get("enterpriseToRevenue", None),
    "EV_EBITDA": data.get("enterpriseToEbitda", None)
}

valuation_data_df = pd.DataFrame([valuation_data])

create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {required_env_vars['SNOWFLAKE_SCHEMA']}.NVIDIA_VALUATION (
    "Market_Cap_intraday" FLOAT,
    "Enterprise_Value" FLOAT,
    "Trailing_PE" FLOAT,
    "Forward_PE" FLOAT,
    "PEG_Ratio" FLOAT,
    "Price_Sales" FLOAT,
    "EV_Revenue" FLOAT,
    "EV_EBITDA" FLOAT,
    LOAD_TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
"""
conn.cursor().execute(create_table_sql)

# Write data with auto-mapping (ensure correct DB/Schema)
success, nchunks, nrows, _ = write_pandas(
    conn=conn,
    df=valuation_data_df,
    table_name='NVIDIA_VALUATION',
    database=required_env_vars['SNOWFLAKE_DB'],  # Ensure consistency
    schema=required_env_vars['SNOWFLAKE_SCHEMA'],
    overwrite=False,
    auto_create_table=False
)

print(f"✅ Inserted {nrows} rows across {nchunks} chunks.")

cur.close()
conn.close()
print("🔻 Connection closed.")