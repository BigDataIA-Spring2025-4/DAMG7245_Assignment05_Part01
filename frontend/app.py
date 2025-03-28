import json
import time
import streamlit as st
import requests
import os
import base64
import pandas as pd
import snowflake.connector
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL")

agents = {
    "Snowflake Agent": "snowflake_query",
    "RAG Agent": "vector_search",
    "Web Search Agent": "web_search",
}

if 'messages' not in st.session_state:
    st.session_state.messages = []
if "file_upload" not in st.session_state:
    st.session_state.file_upload = None

def connect_to_snowflake():
    """Establish connection to Snowflake."""
    try:
        conn = snowflake.connector.connect(
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            role=os.getenv('SNOWFLAKE_ROLE'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DB'),
            schema=os.getenv('SNOWFLAKE_SCHEMA')
        )
        return conn
    except Exception as e:
        st.error(f"Connection to Snowflake failed: {e}")
        return None

def execute_snowflake_query(query):
    """Execute a query against Snowflake and return results as a DataFrame."""
    conn = connect_to_snowflake()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute(query)
            df = cur.fetch_pandas_all()
            cur.close()
            conn.close()
            return df
        except Exception as e:
            st.error(f"Query execution failed: {e}")
            conn.close()
    return pd.DataFrame()  # Return empty DataFrame if query fails

def get_df(analysis_type):
    queries = {
        'stock_performance': """
        SELECT DATE, OPEN, HIGH, LOW, CLOSE, VOLUME,
        ROUND((CLOSE - LAG(CLOSE) OVER (ORDER BY DATE)) / LAG(CLOSE) OVER (ORDER BY DATE) * 100, 2) AS DAILY_RETURN_PERCENT
        FROM NVDA_HISTORICAL_5Y
        """,
        'financial_summary': """
        SELECT DATE, TOTAL_REVENUE, NET_INCOME, EBITDA, DILUTED_EPS
        FROM NVDA_FINANCIALS
        """,
        'balance_sheet': """
        SELECT DATE, TOTAL_ASSETS, TOTAL_LIABILITIES_NET_MINORITY_INTEREST, STOCKHOLDERS_EQUITY, WORKING_CAPITAL
        FROM NVDA_BALANCE_SHEET
        """,
        'earnings_analysis': """
        SELECT DATE, TOTAL_REVENUE, NET_INCOME, BASIC_EPS, DILUTED_EPS, GROSS_PROFIT
        FROM NVDA_FINANCIALS
        """
    }
    return {key: execute_snowflake_query(query) for key, query in queries.items() if key in analysis_type}

def build_report(output: dict):
    research_steps = "\n".join([f"- {r}" for r in output.get("research_steps", [])])
    sources = "\n".join([f"- {s}" for s in output.get("sources", [])])
    
    report = f"""
    COMPREHENSIVE RESEARCH REPORT
    -----------------------------
    RESEARCH STEPS
    --------------
    {research_steps}
    
    HISTORICAL PERFORMANCE ANALYSIS
    -------------------------------
    {output.get("historical_performance", "")}
    
    FINANCIAL VALUATION METRICS
    ---------------------------
    {output.get("analysis_type", "")}
    """
    st.markdown(report)
    
    dataframes = get_df(output.get("analysis_type", ""))
    if "stock_performance" in dataframes:
        df = dataframes["stock_performance"]
        st.subheader("Stock Price Performance")
        st.write(df)
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["DATE"],
            open=df["OPEN"], high=df["HIGH"], low=df["LOW"], close=df["CLOSE"],
            name="Price"
        ))
        fig.update_layout(title="Stock Price Movement", xaxis_title="Date", yaxis_title="Price ($)", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)
        
        fig_returns = px.bar(df, x="DATE", y="DAILY_RETURN_PERCENT", color=df["DAILY_RETURN_PERCENT"].apply(lambda x: "Positive" if x >= 0 else "Negative"), color_discrete_map={"Positive": "green", "Negative": "red"})
        st.subheader("Daily Returns (%)")
        st.plotly_chart(fig_returns)
    
    remaining_report = f"""
    REAL-TIME INDUSTRY INSIGHTS
    ---------------------------
    {output.get("industry_insights", "")}
    
    SUMMARY
    ---------------------------
    {output.get("summary", "")}
    
    SOURCES
    -------
    {sources}
    """
    st.markdown(remaining_report)

def main():
    st.set_page_config(page_title="Research NVIDIA", layout="wide", initial_sidebar_state="expanded")
    st.title("NVIDIA Financial Reports Analysis")
    
    st.sidebar.header("Main Menu")
    year = str(st.sidebar.selectbox('Year', ["2025", "2024", "2023", "2022", "2021"]))
    quarter = st.sidebar.multiselect('Quarter:', ["Q1", "Q2", "Q3", "Q4"], default=None)
    agent_selections = st.sidebar.multiselect('Agents:', list(agents.keys()), default=["Web Search Agent"])
    selected_agents = [agents[key] for key in agent_selections]
    
    query = st.sidebar.text_input("Enter query here:")
    if query:
        st.session_state.messages.clear()
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(f"{API_URL}/query_research_agent", json={"year": year, "quarter": quarter, "query": query, "tools": selected_agents})
                    if response.status_code == 200:
                        answer = response.json()["answer"]
                        build_report(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()