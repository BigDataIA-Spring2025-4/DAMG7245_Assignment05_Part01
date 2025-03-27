import json, time
import streamlit as st
import requests, os, base64
from io import StringIO
from dotenv import load_dotenv
load_dotenv()

API_URL=os.getenv("API_URL")
AIRFLOW_API_URL =os.getenv("AIRFLOW_API_URL")

API_URL = "http://127.0.0.1:8000"

agents = {
    "Snowflake Agent": "snowflake",
    "RAG Agent": "vector_search",
    "Web Search Agent": "web_search",
}

if 'messages' not in st.session_state:
    st.session_state.messages = []
if "file_upload" not in st.session_state:
    st.session_state.file_upload = None

def trigger_airflow_dag():
    st.write("Now triggering airflow dag...")
    year = st.selectbox("Select Year", range(2025, 2020,-1))
    trigger = st.button("Trigger Airflow DAG", use_container_width=True, icon = "🚀")
    if trigger:
        # Payload for triggering the DAG
        payload = {
            "conf": {
                "year": str(year)
            }
        }
        # Trigger the DAG via Airflow REST API
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
    # Set up navigation
    st.sidebar.header("Main Menu") 
    task = st.sidebar.selectbox("Select Task", ["Trigger Airflow", "NVIDIA Research"])
    if task == "Trigger Airflow":
        trigger_airflow_dag()
    elif task == "NVIDIA Research":
        year = str(st.sidebar.selectbox('Year', ["2025", "2024", "2023", "2022", '2021']))
        quarter = st.sidebar.multiselect('Quarter:', ["Q1", "Q2", "Q3", "Q4", ''], default=None)
        agent_selections = st.sidebar.multiselect('Agents:', list(agents.keys()), default=["Web Search Agent"])
        selected_agents = []
        for key in agent_selections:
            selected_agents.append(agents[key])

        st.sidebar.text("Enter query here:")
        query = st.sidebar.chat_input(placeholder = "Write your query here...")
        if query:
            st.session_state.messages.clear()
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # time.sleep(2)
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
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        else:
                            error_message = f"Error: {response.text}"
                            st.error(error_message)
                            st.session_state.messages.append({"role": "assistant", "content": error_message})
                    except Exception as e:
                        error_message = f"Error: {str(e)}"
                        # st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                # st.write(st.session_state.messages)
            
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
                # st.subheader(data["message"])
                st.success("Document Processed Successfully!")
                progress_bar.progress(100)
                progress_text.empty()
                progress_bar.empty()    
                return data["file_name"], data["scraped_content"]
            else:
                st.error("Server not responding.")
        except:
            st.error("An error occurred while processing the PDF.")
    
    
if __name__ == "__main__":
# Set page configuration
    st.set_page_config(
        page_title="Research NVDIA",
        layout="wide",
        initial_sidebar_state="expanded"
    )    
    main()