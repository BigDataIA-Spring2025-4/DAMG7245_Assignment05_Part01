import os
from fastapi import FastAPI, HTTPException
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


@app.post("/query_research_agent")
def query_nvdia_documents(request: NVDIARequest):
    try:
        year = request.year
        quarter = request.quarter
        query = request.query
        tools = request.tools

        print("Tools selected are:", tools)

        runnable = run_agents(tools, year, quarter)
        out = runnable.invoke({ 
            "input": query, 
            "chat_history": [], 
            "year": year, 
            "quarter": quarter 
        })
        print(out)
        answer = out["intermediate_steps"][-1].tool_input
        
        return {
            # "answer": year + quarter[0] + parser + chunk_strategy + vector_store + query,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")