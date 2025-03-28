import csv
import re
import json, time
import streamlit as st
import requests, os, base64
import pandas as pd
from io import StringIO
import plotly.graph_objects as go

from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv("API_URL")
AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL")

API_URL = "http://127.0.0.1:8000"

agents = {
    "Snowflake Agent": "snowflake_query",
    "RAG Agent": "vector_search",
    "Web Search Agent": "web_search",
}

if 'messages' not in st.session_state:
    st.session_state.messages = []
if "file_upload" not in st.session_state:
    st.session_state.file_upload = None

def trigger_airflow_dag():
    st.write("Now triggering airflow dag...")
    year = st.selectbox("Select Year", range(2025, 2020, -1))
    trigger = st.button("Trigger Airflow DAG", use_container_width=True, icon="🚀")
    if trigger:
        payload = {
            "conf": {
                "year": str(year)
            }
        }
        response = requests.post(
            f"{AIRFLOW_API_URL}/api/v1/dags/nvidia_financial_docs_scraper_and_loader/dagRuns",
            json=payload,
            auth=(f"{os.getenv('AIRFLOW_USER')}", f"{os.getenv('AIRFLOW_PASSCODE')}")
        )
        if response.status_code == 200:
            st.success("DAG triggered successfully!")
        else:
            st.error(f"Failed to trigger DAG: {response.text}")

def main():
    st.title("NVDIA Financial Reports Analysis")
    st.sidebar.header("Main Menu") 
    task = st.sidebar.selectbox("Select Task", ["Trigger Airflow", "NVIDIA Research"])
    
    if task == "Trigger Airflow":
        trigger_airflow_dag()
    elif task == "NVIDIA Research":
        year = str(st.sidebar.selectbox('Year', ["2025", "2024", "2023", "2022", "2021"]))
        quarter = st.sidebar.multiselect('Quarter:', ["Q1", "Q2", "Q3", "Q4", ''], default=None)
        agent_selections = st.sidebar.multiselect('Agents:', list(agents.keys()), default=["Web Search Agent"])
        selected_agents = [agents[key] for key in agent_selections]

        st.sidebar.text("Enter query here:")
        query = st.sidebar.chat_input(placeholder="Write your query here...")
        if query:
            st.session_state.messages.clear()
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/query_research_agent",
                            json={
                                "year": year,
                                "quarter": quarter,
                                "query": query,
                                "tools": selected_agents
                            }
                        )
                        if response.status_code == 200:
                            answer = response.json()["answer"]
                            build_report(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                                
                        if "snowflake_agent" in selected_agents:
                            try:
                                df = None
                                if isinstance(answer, dict) and 'snowflake_results' in answer:
                                    snowflake_data = answer['snowflake_results']
                                    if snowflake_data:
                                        # Split the string into lines and remove any empty lines
                                        lines = [line.strip() for line in snowflake_data.strip().splitlines() if line.strip()]
                                        
                                        # Extract headers from the first line
                                        headers = lines[0].split()
                                        
                                        # Process data rows - use a more robust approach
                                        data_rows = []
                                        for line in lines[1:]:
                                            # Skip lines that don't contain data
                                            if not line or line.isspace():
                                                continue
                                            
                                            # Split the line by whitespace
                                            # The strip and split approach handles variable spacing
                                            parts = line.strip().split()
                                            
                                            # Ensure we have the right number of columns
                                            # If the first element is an index, remove it
                                            if len(parts) > len(headers) and parts[0].isdigit():
                                                parts = parts[1:]
                                            
                                            # Pad shorter rows or truncate longer rows
                                            if len(parts) < len(headers):
                                                parts.extend([''] * (len(headers) - len(parts)))
                                            elif len(parts) > len(headers):
                                                parts = parts[:len(headers)]
                                            
                                            data_rows.append(parts)
                                        
                                        # Create DataFrame
                                        df = pd.DataFrame(data_rows, columns=headers)
                                        
                                        # Convert columns to appropriate types
                                        if "DATE" in df.columns:
                                            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
                                        
                                        numeric_cols = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "DIVIDENDS", "STOCKSPLITS"]
                                        for col in numeric_cols:
                                            if col in df.columns:
                                                df[col] = pd.to_numeric(df[col], errors="coerce")
                                        
                                        # Create chart
                                        if "DATE" in df.columns and "CLOSE" in df.columns:
                                            df = df.dropna(subset=["DATE", "CLOSE"])
                                            df = df.sort_values("DATE")
                                            st.subheader("NVIDIA Closing Price Over Time")
                                            st.line_chart(df.set_index("DATE")["CLOSE"])
                                        else:
                                            st.warning(f"Required columns missing. Available columns: {df.columns.tolist()}")
                                    else:
                                        st.warning("No data returned from Snowflake agent.")
                            except Exception as e:
                                st.error(f"Error processing chart data: {str(e)}")
                                import traceback
                                st.write(traceback.format_exc())

                        else:
                            error_message = f"Error: {response.text}"
                            st.error(error_message)
                            st.session_state.messages.append({"role": "assistant", "content": error_message})
                    except Exception as e:
                        error_message = f"Error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                # st.write(st.session_state.messages)


import pandas as pd
import snowflake.connector
import plotly.express as px

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
            if conn:
                conn.close()
    return pd.DataFrame()  # Return empty DataFrame if query fails

def get_df(analysis_type):
    df = {}
    queries = {
        'stock_performance': """
        SELECT 
            DATE, 
            OPEN, 
            HIGH, 
            LOW, 
            CLOSE, 
            VOLUME,
            ROUND((CLOSE - LAG(CLOSE) OVER (ORDER BY DATE)) / LAG(CLOSE) OVER (ORDER BY DATE) * 100, 2) AS DAILY_RETURN_PERCENT
        FROM NVDA_HISTORICAL_5Y
        """,
        
        'financial_summary': """
        SELECT 
            DATE, 
            TOTAL_REVENUE, 
            NET_INCOME, 
            EBITDA, 
            DILUTED_EPS
        FROM NVDA_FINANCIALS
        """,
        
        'balance_sheet': """
        SELECT 
            DATE, 
            TOTAL_ASSETS, 
            TOTAL_LIABILITIES_NET_MINORITY_INTEREST, 
            STOCKHOLDERS_EQUITY, 
            WORKING_CAPITAL
        FROM NVDA_BALANCE_SHEET
        """,
        
        'earnings_analysis': """
        SELECT 
            DATE, 
            TOTAL_REVENUE, 
            NET_INCOME, 
            BASIC_EPS, 
            DILUTED_EPS, 
            GROSS_PROFIT
        FROM NVDA_FINANCIALS
        """
    }
    if "stock_performance" in analysis_type:
        # query.append(queries["stock_performance"])
        # df_stock = execute_snowflake_query(queries["stock_performance"])
        df["stock_performance"] = execute_snowflake_query(queries["stock_performance"])

    if "financial_summary" in analysis_type:
        # query.append(queries["financial_summary"])
        df["financial_summary"] = execute_snowflake_query(queries["financial_summary"])

    if "balance_sheet" in analysis_type:
        # query.append(queries["balance_sheet"])
        # df_bal = execute_snowflake_query(queries["balance_sheet"])
        df["balance_sheet"] = execute_snowflake_query(queries["balance_sheet"])

    if "earnings_analysis" in analysis_type:
        # query.append(queries["earnings_analysis"])
        df["earnings_analysis"] = execute_snowflake_query(queries["earnings_analysis"])
    

    return df

def generate_stock_performance_graphs(dataframe):
    print(dataframe)



def build_report(output: dict):
    research_steps = output.get("research_steps", "") 
    if isinstance(research_steps, list): 
        research_steps = "\n".join([f"- {r}" for r in research_steps]) 
     
    sources = output.get("sources", "") 
    if isinstance(sources, list): 
        sources = "\n".join([f"- {s}" for s in sources]) 
    
    # Get the financial query
    financial_query = output.get("financial_queries", "")
    
    # Display the main report
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
    st.write(dataframes)

    if "stock_performance" in dataframes:
        # Creating a stock price visualization
        st.subheader("Stock Price Performance")
        
        df = dataframes.get("stock_performance", "")

        # Extract the necessary columns for plotting
        chart_data = df[["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "DAILY_RETURN_PERCENT"]]
        
        st.write(chart_data)

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
        st.plotly_chart(fig)
        
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
        
        st.plotly_chart(fig_returns)
        
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
        
        st.plotly_chart(fig_volume)


    # Handle financial visualization section


    
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

if __name__ == "__main__":
    st.set_page_config(
        page_title="Research NVDIA",
        layout="wide",
        initial_sidebar_state="expanded"
    )    
    main()
