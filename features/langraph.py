from typing import TypedDict, Annotated, Optional, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
import json
import plotly.graph_objects as go

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import ToolCall, ToolMessage
from langchain_openai import ChatOpenAI
from serpapi import GoogleSearch

from functools import partial

import os
from dotenv import load_dotenv

from features.pinecone_index import query_pinecone
from features.snowflake_agent import snowflake_query_agent

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")


required_env_vars = {
    'SNOWFLAKE_ACCOUNT': os.getenv('SNOWFLAKE_ACCOUNT'),
    'SNOWFLAKE_USER': os.getenv('SNOWFLAKE_USER'),
    'SNOWFLAKE_PASSWORD': os.getenv('SNOWFLAKE_PASSWORD'),
    'SNOWFLAKE_ROLE': os.getenv('SNOWFLAKE_ROLE'),
    'SNOWFLAKE_DB': os.getenv('SNOWFLAKE_DB'),
    'SNOWFLAKE_WAREHOUSE': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'SNOWFLAKE_SCHEMA': os.getenv('SNOWFLAKE_SCHEMA')
}


## Creating the Agent State ##
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    year: Optional[str]
    quarter: Optional[List]



@tool("vector_search")
def vector_search(query: str, year: str = None, quarter: list = None):
    """Searches for the most relevant vector in the Pinecone index."""
    # query = "What is the revenue of Nvidia?"
    # year = "2025"
    # quarter = ['Q4', 'Q1']
    print("Reached Vector search 1")
    top_k = 10
    chunks = query_pinecone(query, top_k, year = year, quarter = quarter)
    contexts = "\n---\n".join(
        {chr(10).join([f'Chunk {i+1}: {chunk}' for i, chunk in enumerate(chunks)])}
    )
    return contexts

@tool("web_search")
def web_search(query: str):
    """Finds general knowledge information using Google search. Can also be used
    to augment more 'general' knowledge to a previous specialist query."""
    # Web Search Tool
    serpapi_params = {
        "api_key": SERPAPI_KEY,  # <-- Add your SerpAPI key
        "engine": "google",  # Specifies Google Search Engine
    }
    search = GoogleSearch({
        **serpapi_params,
        "q": query,
        "num": 5
    })
    results = search.get_dict()["organic_results"]
    contexts = "\n---\n".join(
        ["\n".join([x["title"], x["snippet"], x["link"]]) for x in results]
    )
    return contexts

# LangGraph tool integration
import pandas as pd
from datetime import datetime

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle pandas Timestamp and other non-serializable objects"""
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

@tool("snowflake_query")
def snowflake_query(
    query: str = None, 
    analysis_type: str = 'financial_summary', 
    year: str = None, 
    quarter: List[str] = None
):
    """
    LangGraph compatible Snowflake query tool
    
    Args match snowflake_query_agent function
    """
    result = snowflake_query_agent(
        query=query, 
        analysis_type=analysis_type, 
        year=year, 
        quarter=quarter
    )
    
    # Convert Plotly figures to JSON string representation for each visualization
    if 'visualizations' in result:
        result['visualizations'] = {
            k: fig.to_json() for k, fig in result['visualizations'].items()
        }
    
    # Return as a string representation that includes visualization data
    return str(result)


@tool("final_answer")
def final_answer(
    research_steps: str,
    historical_performance: str,
    financial_analysis: str,
    industry_insights: str,
    summary: str,
    sources: str,
    financial_visualizations: dict = None
):
    """
    Returns a comprehensive research report combining data from all agents.
    
    Args:
    -research_steps: List points for each step that was taken
    to research your report. Format: Tool Name: step taken  (For each tool step)
    - historical_performance: Detailed Analysis from RAG/vector_search agent. Write at least 2 paragraphs
    - financial_analysis: 1) Write at least 1 or 2 paragraphs using previous response and Snowflake agent's financial summary and metrics
                            2)Display table , if possible, in markdown format for the queries produced (for the columns that have data).If snowflake tool not used, then a one sentence from other tool output
    - financial_visualizations: snowflake_visualizations -> Plotly figures as dict
    - industry_insights: Detailed Real-time trends from web search agent, Write at least 3 paragraphs, if web search tool not used, then a one sentence from other tool output
    - summary: A Fully detailed summary of at least 4 big paragraphs using all responses from all tools
    - sources: If web search tool used, List of all referenced sources. Give links where possible
    
    Returns:
    Structured dictionary with complete research report components
    """
    if type(research_steps) is list:
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    if type(sources) is list:
        sources = "\n".join([f"- {s}" for s in sources])
    
    # Ensure financial_visualizations is always a dict, never None
    if financial_visualizations is None:
        financial_visualizations = {}
    
    report = {
        "research_steps": research_steps,
        "historical_performance": historical_performance,
        "financial_analysis": financial_analysis,
        "financial_visualizations": financial_visualizations,
        "industry_insights": industry_insights,
        "summary": summary,
        "sources": sources
    }
    
    return report

def init_research_agent(tool_keys, year=None, quarter=None):
    tool_str_to_func = {
            "web_search": web_search,
            "vector_search": vector_search,
            "snowflake_query": snowflake_query,
            "final_answer": final_answer
        }
    
    tools = [final_answer]
    for val in tool_keys:
        tools.append(tool_str_to_func[val])

    ## Designing Agent Features and Prompt ##
    system_prompt = f"""You are NVIDIA research agent that has multiple tools available for research and NVIDIA information retrieval.
    Given the user's query you must decide what to do with it based on the
    list of tools provided to you.

    Context:
    - Year: {year or 'Not specified'}
    - Quarter: {quarter or 'Not specified'}

    Use all the Tools available at least once.But not more than 3 times.
    If you see that a tool has been used (in the scratchpad) with a particular
    query, do NOT use that same tool with the same query again. Also, do NOT use
    any tool more than twice (ie, if the tool appears in the scratchpad twice, do
    not use it again).

    You should aim to collect information from a diverse range of sources regarding NVIDIA before
    providing the answer to the user. Once you have collected relevant information
    to answer the user's question (stored in the scratchpad) use the final_answer
    tool."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("assistant", "scratchpad: {scratchpad}"),
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        temperature=0
    )

    def create_scratchpad(intermediate_steps: list[AgentAction]):
        research_steps = []
        for i, action in enumerate(intermediate_steps):
            if action.log != "TBD":
                # this was the ToolExecution
                research_steps.append(
                    f"Tool: {action.tool}, input: {action.tool_input}\n"
                    f"Output: {action.log}"
                )
        return "\n---\n".join(research_steps)

    oracle = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "scratchpad": lambda x: create_scratchpad(
                intermediate_steps=x["intermediate_steps"]
            ),
        }
        | prompt
        | llm.bind_tools(tools, tool_choice="any")
    )
    return oracle

## Router and Parent Agent functions
def run_oracle(state: AgentState, oracle):
    print("run_oracle")
    print(f"intermediate_steps: {state['intermediate_steps']}")
    
    out = oracle.invoke(state)
    tool_name = out.tool_calls[0]["name"]
    tool_args = out.tool_calls[0]["args"]
    
    # Transfer visualization data to final_answer
    if tool_name == "final_answer":
        if "snowflake_data" in state and "visualizations" in state["snowflake_data"]:
            print(f"Adding visualizations to final_answer: {list(state['snowflake_data']['visualizations'].keys())}")
            tool_args["financial_visualizations"] = state["snowflake_data"]["visualizations"]
        else:
            # Ensure the parameter exists even if empty
            tool_args["financial_visualizations"] = {}
    
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log="TBD"
    )
    
    return {
        **state,
        "intermediate_steps": [action_out]
    }
    
def router(state: AgentState):
    # Add a failsafe for maximum iterations
    if len(state.get("intermediate_steps", [])) > 10:
        print("Maximum iterations reached, forcing final_answer")
        return "final_answer"
    
    # Return the tool name to use
    if isinstance(state["intermediate_steps"], list) and state["intermediate_steps"]:
        if state["intermediate_steps"][-1].tool:
            return state["intermediate_steps"][-1].tool
    
    # Default to final_answer if anything goes wrong
    print("Router defaulting to final_answer")
    return "final_answer"
    


def run_tool(state: AgentState):
    tool_str_to_func = {
        "web_search": web_search,
        "vector_search": vector_search,
        "snowflake_query": snowflake_query,
        "final_answer": final_answer
    }
    
    # Get tool name and arguments
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input

    if tool_name in ["vector_search"]:
        tool_args = {
            **tool_args,
            "year": state.get("year"),
            "quarter": state.get("quarter")
        }
    
    print(f"{tool_name}.invoke(input={tool_args})")
    
    # Run tool
    out = tool_str_to_func[tool_name].invoke(input=tool_args)
    
    # Special handling for snowflake_query
    if tool_name == "snowflake_query":
        try:
            # Convert string to dict if needed
            if isinstance(out, str):
                import json
                import ast
                
                # Try to parse the output string
                try:
                    # First try json.loads
                    parsed_out = json.loads(out)
                except:
                    try:
                        # Then try ast.literal_eval
                        parsed_out = ast.literal_eval(out)
                    except:
                        # If both fail, use regex to extract visualizations
                        import re
                        
                        # Initialize empty storage
                        parsed_out = {"visualizations": {}}
                        
                        # Try to find visualization data using regex
                        viz_match = re.search(r"'visualizations':\s*({.*?})", out, re.DOTALL)
                        if viz_match:
                            viz_str = viz_match.group(1)
                            try:
                                # Try to parse the visualizations
                                viz_data = ast.literal_eval(viz_str)
                                parsed_out["visualizations"] = viz_data
                            except:
                                print("Failed to parse visualizations with ast.literal_eval")
            else:
                parsed_out = out
                
            # Store visualizations in state if found
            if parsed_out and isinstance(parsed_out, dict) and "visualizations" in parsed_out:
                if "snowflake_data" not in state:
                    state["snowflake_data"] = {}
                
                state["snowflake_data"]["visualizations"] = parsed_out["visualizations"]
        except Exception as e:
            print(f"Error processing visualization data: {e}")
    
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(out)
    )
    
    return {
        **state,
        "intermediate_steps": [action_out]
    }


## Langraph - Designing the Graph
def create_graph(research_agent, year=None, quarter=None):
    tools=[
        vector_search,
        snowflake_query,
        web_search,
        final_answer
    ]

    # Create the graph with a higher recursion limit
    graph = StateGraph(AgentState)
    

    # Pass state to all functions that require it
    graph.add_node("oracle", partial(run_oracle, oracle=research_agent))
    graph.add_node("web_search", run_tool)
    graph.add_node("vector_search", run_tool)
    graph.add_node("snowflake_query", run_tool)
    graph.add_node("final_answer", run_tool)

    graph.set_entry_point("oracle")

    # Add conditional edges
    graph.add_conditional_edges(
        source="oracle",
        path=router,
    )

    # Create edges from each tool back to the oracle
    for tool_obj in tools:
        if tool_obj.name != "final_answer":
            graph.add_edge(tool_obj.name, "oracle")

    # If anything goes to final answer, it must then move to END
    graph.add_edge("final_answer", END)

    # Add a timeout
    runnable = graph.compile()  # Add a 120-second timeout
    return runnable

def run_agents(tool_keys, year=None, quarter=None):
    research_agent = init_research_agent(tool_keys, year, quarter)
    runnable = create_graph(research_agent, year, quarter)
    return runnable
