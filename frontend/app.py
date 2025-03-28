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
            
def convert_PDF_to_markdown(file_upload, parser):    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    progress_text.text("Uploading file...")
    progress_bar.progress(20)

    if file_upload is not None:
        bytes_data = file_upload.read()
        base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
        
        progress_text.text("Sending file for processing...")
        progress_bar.progress(50)

        response = requests.post(f"{API_URL}/upload_pdf", json={"file": base64_pdf, "file_name": file_upload.name, "parser": parser})
        
        progress_text.text("Processing document...")
        progress_bar.progress(75)
        
        try:
            if response.status_code == 200:
                data = response.json()
                progress_text.text("Finalizing output...")
                st.success("Document Processed Successfully!")
                progress_bar.progress(100)
                progress_text.empty()
                progress_bar.empty()    
                return data["file_name"], data["scraped_content"]
            else:
                st.error("Server not responding.")
        except:
            st.error("An error occurred while processing the PDF.")


def build_report(output: dict):
    research_steps = output.get("research_steps", "")
    if isinstance(research_steps, list):
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    
    sources = output.get("sources", "")
    if isinstance(sources, list):
        sources = "\n".join([f"- {s}" for s in sources])
    
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
{output.get("financial_analysis", "")}

FINANCIAL VISUALIZATIONS
------------------------
"""
    st.markdown(report)
    
    financial_visualizations = output.get("financial_visualizations", {})
    if financial_visualizations:
        for title, fig_json_string in financial_visualizations.items():
            try:
                fig_json = json.loads(fig_json_string)
                fig = go.Figure(fig_json)
                st.subheader(title)
                st.plotly_chart(fig)
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON for {title}: {str(e)}")
            except Exception as e:
                st.error(f"Error rendering {title}: {str(e)}")

    report = f"""
REAL-TIME INDUSTRY INSIGHTS
---------------------------
{output.get("industry_insights", "")}

SUMMARY
---------------------------
{output.get("summary", "")}

SOURCES
-------
{output.get("sources", "")}
"""
    st.markdown(report)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Research NVDIA",
        layout="wide",
        initial_sidebar_state="expanded"
    )    
    main()
