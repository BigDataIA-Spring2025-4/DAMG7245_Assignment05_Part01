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
# AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL")

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
    research_steps = output.get("research_steps", "") 
    if isinstance(research_steps, list): 
        research_steps = "\n".join([f"- {r}" for r in research_steps]) 
     
    sources = output.get("sources", "") 
    if isinstance(sources, list): 
        sources = "\n".join([f"- {s}" for s in sources]) 
    
    # Get the financial query
    financial_query = output.get("financial_queries", "")
        
    report = f"""
    COMPREHENSIVE RESEARCH REPORT:
    ---------------------------------------------------

    SUMMARY
    ---------------------------
    {output.get("summary", "")}
    
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
        generate_stock_performance_chart(dataframes)
    if "financial_summary" in dataframes:
        generate_financial_summary_chart(dataframes)
    if "balance_sheet" in dataframes:
        generate_balance_sheet_chart(dataframes)
    if "earnings_analysis" in dataframes:
        generate_earnings_analysis_chart(dataframes)
    
    remaining_report = f"""
    
    REAL-TIME INDUSTRY INSIGHTS
    ---------------------------
    {output.get("industry_insights", "")}
    
    RESEARCH STEPS
    --------------
    {research_steps}
    
    SOURCES
    -------
    {sources}
    """
    st.markdown(remaining_report)

def generate_stock_performance_chart(dataframes):
    st.subheader("Stock Price Performance")        
    df = dataframes.get("stock_performance", "")

    # Extract the necessary columns for plotting
    chart_data = df[["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "DAILY_RETURN_PERCENT"]]
    # Create a price chart
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=chart_data["DATE"],
        open=chart_data["OPEN"], 
        high=chart_data["HIGH"],
        low=chart_data["LOW"], 
        close=chart_data["CLOSE"],
        name="Price"
    ))
    
    # Customize the layout
    fig.update_layout(
        title="Stock Price Movement",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        height=500,
        width=800
    )
    
    # Display the figure
    st.plotly_chart(fig, key = "stock_price_chart", use_container_width=True)
    
    # Add a daily returns chart
    st.subheader("Daily Returns (%)")
    fig_returns = px.bar(
        chart_data, 
        x="DATE", 
        y="DAILY_RETURN_PERCENT",
        color=chart_data["DAILY_RETURN_PERCENT"].apply(lambda x: "Positive" if x >= 0 else "Negative"),
        color_discrete_map={"Positive": "green", "Negative": "red"},
        labels={"DATE": "Date", "DAILY_RETURN_PERCENT": "Daily Return (%)"}
    )
    
    fig_returns.update_layout(
        height=300,
        width=800
    )
    
    st.plotly_chart(fig_returns, key = "daily_returns_chart", use_container_width=True)
    
    # Add volume chart
    st.subheader("Trading Volume")
    fig_volume = px.bar(
        chart_data,
        x="DATE",
        y="VOLUME",
        labels={"DATE": "Date", "VOLUME": "Volume"}
    )
    
    fig_volume.update_layout(
        height=300,
        width=800
    )
    
    st.plotly_chart(fig_volume, key = "volume_chart", use_container_width=True)

def generate_financial_summary_chart(dataframes):
    st.subheader("Financial Summary")

    # Fetch data
    df = dataframes.get("financial_summary", None)
    if df is None or df.empty:
        st.warning("No financial data available.")
        return

    # Ensure DATE is sorted for proper visualization
    df = df.sort_values(by="DATE")

    # Revenue and Net Income Side-by-Side Bar Chart
    st.subheader("Total Revenue & Net Income Over Time")
    fig_revenue = px.bar(df, x="DATE", y=["TOTAL_REVENUE", "NET_INCOME"], 
                         labels={"value": "Amount ($M)", "DATE": "Date"},
                         title="Total Revenue vs Net Income",
                         barmode="group",  # Side-by-side bars
                         color_discrete_map={"TOTAL_REVENUE": "blue", "NET_INCOME": "orange"})  

    fig_revenue.update_layout(height=400, width=800, xaxis=dict(tickmode='auto', nticks=10))
    st.plotly_chart(fig_revenue, key = "revenue_financial_chart", use_container_width=True)

    # EBITDA Plot
    st.subheader("EBITDA Over Time")
    fig_ebitda = px.bar(df, x="DATE", y="EBITDA", 
                        labels={"EBITDA": "EBITDA ($M)", "DATE": "Date"},
                        title="EBITDA Trend", color="EBITDA", color_continuous_scale="blues")

    fig_ebitda.update_layout(height=300, width=800, xaxis=dict(tickmode='auto', nticks=10))
    st.plotly_chart(fig_ebitda, key = "ebitda_financial_chart", use_container_width=True)

    # EPS Plot
    st.subheader("Diluted EPS Over Time")
    fig_eps = px.line(df, x="DATE", y="DILUTED_EPS", 
                      labels={"DILUTED_EPS": "EPS ($)", "DATE": "Date"},
                      title="Diluted Earnings Per Share (EPS) Trend")

    fig_eps.update_traces(mode="markers+lines", line=dict(color="green"))
    fig_eps.update_layout(height=300, width=800)
    st.plotly_chart(fig_eps, key = "eps_financial_chart", use_container_width=True)
        
def generate_balance_sheet_chart(dataframes):
    st.subheader("Balance Sheet Overview")

    # Fetch data
    df = dataframes.get("balance_sheet", None)
    if df is None or df.empty:
        st.warning("No balance sheet data available.")
        return

    # Ensure DATE is sorted for proper visualization
    df = df.sort_values(by="DATE")

    # Total Assets vs Total Liabilities (Side-by-Side Bar Chart)
    st.subheader("Total Assets vs Total Liabilities Over Time")
    fig_assets_liabilities = px.bar(df, x="DATE", 
                                    y=["TOTAL_ASSETS", "TOTAL_LIABILITIES_NET_MINORITY_INTEREST"], 
                                    labels={"value": "Amount ($M)", "DATE": "Date"},
                                    title="Total Assets vs Total Liabilities",
                                    barmode="group",  # Side-by-side bars
                                    color_discrete_map={"TOTAL_ASSETS": "blue", "TOTAL_LIABILITIES_NET_MINORITY_INTEREST": "red"})

    fig_assets_liabilities.update_layout(height=400, width=800, xaxis=dict(tickmode='auto', nticks=10))
    st.plotly_chart(fig_assets_liabilities, key = "assets_liabilities_balance_sheet_chart", use_container_width=True)

    # Stockholders' Equity Trend (Line Chart)
    st.subheader("Stockholders' Equity Over Time")
    fig_equity = px.line(df, x="DATE", y="STOCKHOLDERS_EQUITY",
                         labels={"STOCKHOLDERS_EQUITY": "Equity ($M)", "DATE": "Date"},
                         title="Stockholders' Equity Trend")

    fig_equity.update_traces(mode="markers+lines", line=dict(color="green"))
    fig_equity.update_layout(height=300, width=800)
    st.plotly_chart(fig_equity, key = "equity_balance_sheet_chart", use_container_width=True)

    # Working Capital Trend (Bar Chart)
    st.subheader("Working Capital Over Time")
    fig_working_capital = px.bar(df, x="DATE", y="WORKING_CAPITAL",
                                 labels={"WORKING_CAPITAL": "Working Capital ($M)", "DATE": "Date"},
                                 title="Working Capital Trend",
                                 color="WORKING_CAPITAL", color_continuous_scale="blues")

    fig_working_capital.update_layout(height=300, width=800, xaxis=dict(tickmode='auto', nticks=10))
    st.plotly_chart(fig_working_capital, key = "working_capital_balance_sheet_chart", use_container_width=True)
    
def generate_earnings_analysis_chart(dataframes):
    st.subheader("Earnings Analysis")

    # Fetch data
    df = dataframes.get("earnings_analysis", None)
    if df is None or df.empty:
        st.warning("No earnings data available.")
        return

    # Ensure DATE is sorted for proper visualization
    df = df.sort_values(by="DATE")

    # Total Revenue vs Net Income (Side-by-Side Bar Chart)
    st.subheader("Total Revenue vs Net Income Over Time")
    fig_revenue_income = px.bar(df, x="DATE", 
                                y=["TOTAL_REVENUE", "NET_INCOME"], 
                                labels={"value": "Amount ($M)", "DATE": "Date"},
                                title="Total Revenue vs Net Income",
                                barmode="group",  # Side-by-side bars
                                color_discrete_map={"TOTAL_REVENUE": "blue", "NET_INCOME": "green"})  

    fig_revenue_income.update_layout(height=400, width=800, xaxis=dict(tickmode='auto', nticks=10))
    st.plotly_chart(fig_revenue_income, key = "revenue_income_earnings_chart", use_container_width=True)

    # EPS (Basic & Diluted) Trend (Line Chart)
    st.subheader("Earnings Per Share (EPS) Over Time")
    fig_eps = px.line(df, x="DATE", y=["BASIC_EPS", "DILUTED_EPS"],
                      labels={"value": "EPS ($)", "DATE": "Date"},
                      title="Basic EPS vs Diluted EPS Trend")

    fig_eps.update_traces(mode="markers+lines")
    fig_eps.update_layout(height=300, width=800)
    st.plotly_chart(fig_eps, key = "eps_earnings_chart", use_container_width=True)

    # Gross Profit Trend (Bar Chart)
    st.subheader("Gross Profit Over Time")
    fig_gross_profit = px.bar(df, x="DATE", y="GROSS_PROFIT",
                              labels={"GROSS_PROFIT": "Gross Profit ($M)", "DATE": "Date"},
                              title="Gross Profit Trend",
                              color="GROSS_PROFIT", color_continuous_scale="greens")

    fig_gross_profit.update_layout(height=300, width=800, xaxis=dict(tickmode='auto', nticks=10))
    st.plotly_chart(fig_gross_profit, key = "gross_profit_earnings_chart", use_container_width=True)

def main():
    st.set_page_config(page_title="Research NVIDIA", layout="wide", initial_sidebar_state="expanded")
    st.title("NVIDIA Financial Reports Analysis")
    
    st.sidebar.header("Main Menu")
    year = str(st.sidebar.selectbox('Year', ["2025", "2024", "2023", "2022", "2021"]))
    quarter = st.sidebar.multiselect('Quarter:', ["Q1", "Q2", "Q3", "Q4"], default=None)
    agent_selections = st.sidebar.multiselect('Agents:', list(agents.keys()), default=["Web Search Agent"])
    selected_agents = [agents[key] for key in agent_selections]
    
    st.sidebar.text("Enter your query below:")
    query = st.sidebar.chat_input("Enter query here:")
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