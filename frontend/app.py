import json, time
import streamlit as st
import requests, os, base64
from io import StringIO
import plotly.graph_objects as go

from dotenv import load_dotenv
load_dotenv()

API_URL=os.getenv("API_URL")
AIRFLOW_API_URL =os.getenv("AIRFLOW_API_URL")

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
                            f"{API_URL}/nvidia-research",
                            json={
                                "year": year,
                                "quarter": quarter,
                                "query": query,
                                "tools": selected_agents
                            }
                        )
                        if response.status_code == 200:
                            answer = response.json()
                            render_research_results(answer)
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


def render_research_results(result_data):
    """
    Render research results in Streamlit
    """
    import streamlit as st
    
    # Display research components
    if 'research_steps' in result_data:
        st.markdown("## Research Steps")
        st.markdown(result_data['research_steps'])
    
    if 'historical_performance' in result_data:
        st.markdown("## Historical Performance")
        st.markdown(result_data['historical_performance'])
    
    if 'financial_analysis' in result_data:
        st.markdown("## Financial Analysis")
        st.markdown(result_data['financial_analysis'])
    
    # Render visualizations if available
    if 'financial_visualizations' in result_data and result_data['financial_visualizations']:
        render_base64_visualizations(result_data['financial_visualizations'])
    
    if 'industry_insights' in result_data:
        st.markdown("## Industry Insights")
        st.markdown(result_data['industry_insights'])
    
    if 'summary' in result_data:
        st.markdown("## Summary")
        st.markdown(result_data['summary'])
        
    if 'sources' in result_data:
        st.markdown("## Sources")
        st.markdown(result_data['sources'])


def render_base64_visualizations(visualization_data):
    """
    Render base64 encoded visualizations in Streamlit
    """
    import streamlit as st
    import base64
    from PIL import Image
    import io
    
    st.markdown("## Financial Visualizations")
    
    if not visualization_data:
        st.info("No visualizations available")
        return
    
    for name, base64_str in visualization_data.items():
        try:
            st.subheader(name.replace('_', ' ').title())
            
            # Decode base64 string
            image_bytes = base64.b64decode(base64_str)
            
            # Convert to image and display
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, use_column_width=True)
            
        except Exception as e:
            st.error(f"Error rendering visualization {name}: {str(e)}")
            st.code(base64_str[:100] + "..." if len(base64_str) > 100 else base64_str)



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
# Set page configuration
    st.set_page_config(
        page_title="Research NVDIA",
        layout="wide",
        initial_sidebar_state="expanded"
    )    
    main()