import yfinance as yf
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# Fetch NVIDIA data
ticker = yf.Ticker("NVDA")
data = ticker.info

# Extract valuation metrics
valuation_data = {
    "Market Cap (intraday)": data.get("marketCap", "N/A"),
    "Enterprise Value": data.get("enterpriseValue", "N/A"),
    "Trailing P/E": data.get("trailingPE", "N/A"),
    "Forward P/E": data.get("forwardPE", "N/A"),
    "PEG Ratio": data.get("pegRatio", "N/A"),
    "Price/Sales": data.get("priceToSalesTrailing12Months", "N/A"),
    "Enterprise Value/Revenue": data.get("enterpriseToRevenue", "N/A"),
    "Enterprise Value/EBITDA": data.get("enterpriseToEbitda", "N/A")
}

"""
CREATE TABLE NVIDIA_VALUATION (
METRIC_DATE TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
MARKET_CAP FLOAT,
ENTERPRISE_VALUE FLOAT,
TRAILING_PE FLOAT,
FORWARD_PE FLOAT,
PEG_RATIO FLOAT,
PRICE_TO_SALES FLOAT,
ENTERPRISE_VALUE_TO_REVENUE FLOAT,
ENTERPRISE_VALUE_TO_EBITDA FLOAT
);
"""

# Configure Snowflake connection
conn = snowflake.connector.connect(
    user="YOUR_USER",
    password="YOUR_PASSWORD",
    account="YOUR_ACCOUNT",
    warehouse="YOUR_WAREHOUSE",
    database="YOUR_DB",
    schema="YOUR_SCHEMA"
)

# Insert data
cursor = conn.cursor()
insert_query = """
INSERT INTO NVIDIA_VALUATION (
    MARKET_CAP,
    ENTERPRISE_VALUE,
    TRAILING_PE,
    FORWARD_PE,
    PEG_RATIO,
    PRICE_TO_SALES,
    ENTERPRISE_VALUE_TO_REVENUE,
    ENTERPRISE_VALUE_TO_EBITDA
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

values = (
    valuation_data["Market Cap (intraday)"],
    valuation_data["Enterprise Value"],
    valuation_data["Trailing P/E"],
    valuation_data["Forward P/E"],
    valuation_data["PEG Ratio"],
    valuation_data["Price/Sales"],
    valuation_data["Enterprise Value/Revenue"],
    valuation_data["Enterprise Value/EBITDA"]
)

cursor.execute(insert_query, values)
conn.commit()
cursor.close()
conn.close()