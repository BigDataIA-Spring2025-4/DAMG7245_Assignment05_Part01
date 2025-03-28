import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from features.langraph import run_agents

from dotenv import load_dotenv
load_dotenv()


from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

class NVDIARequest(BaseModel):
    year: str
    quarter: list
    query: str
    tools: list
    
app = FastAPI()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_FILE_INDEX = os.getenv("PINECONE_FILE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

client = OpenAI()

@app.get("/")
def read_root():
    return {"message": "NVDIA Financial Reports Analysis: FastAPI Backend with OpenAI Integration available for user queries..."}


@app.post("/nvidia-research")
async def nvidia_research(request: dict):
    query = request.get('query', '')
    tools = request.get('tools', [])
    year = request.get('year')
    quarter = request.get('quarter')
    
    try:
        # Run your agent
        runnable = run_agents(tools, year, quarter)
        result = runnable.invoke({ 
            "input": query, 
            "chat_history": [], 
            "year": year, 
            "quarter": quarter 
        })
        
        # Get the final answer
        answer = result["intermediate_steps"][-1].tool_input
        
        # Return the answer (with any visualizations now properly included)
        return JSONResponse(content=answer)
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        raise HTTPException(status_code=500, detail=f"Research agent error: {str(e)}")
    

# def build_report(output: dict):
#     research_steps = output["research_steps"]
#     if type(research_steps) is list:
#         research_steps = "\n".join([f"- {r}" for r in research_steps])
#     sources = output["sources"]
#     if type(sources) is list:
#         sources = "\n".join([f"- {s}" for s in sources])
#     return f"""
# INTRODUCTION
# ------------
# {output["introduction"]}

# RESEARCH STEPS
# --------------
# {research_steps}

# REPORT
# ------
# {output["main_body"]}

# CONCLUSION
# ----------
# {output["conclusion"]}

# SOURCES
# -------
# {sources}
# """
# def dict_to_graph():
#     for title, fig_json in report["financial_visualizations"].items():
#         fig = go.Figure(fig_json)  # Reconstruct Plotly figure from JSON
#         st.plotly_chart(fig)
        
# def build_report(output: dict):
#     research_steps = output["research_steps"]
#     if type(research_steps) is list:
#         research_steps = "\n".join([f"- {r}" for r in research_steps])
#     sources = output["sources"]
#     if isinstance(sources, list):
#         sources = "\n".join([f"- {s}" for s in sources])
    
#     report = f"""
# COMPREHENSIVE RESEARCH REPORT
# -----------------------------
# RESEARCH STEPS
# --------------
# {research_steps}

# HISTORICAL PERFORMANCE ANALYSIS
# -------------------------------
# {output["historical_performance"]}

# FINANCIAL VALUATION METRICS
# ---------------------------
# {output["financial_analysis"]}

# FINANCIAL VISUALIZATIONS
# ------------------------
# [Plotly visualizations would be rendered here in Streamlit]

# REAL-TIME INDUSTRY INSIGHTS
# ---------------------------
# {output["industry_insights"]}

# SOURCES
# -------
# {sources}
# """
#     return report