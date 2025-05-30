from typing import TypedDict, Annotated, Optional, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
import json

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
    """
    Searches for the most relevant chunks  in the Pinecone index, 
    which has NVIDIA Quarterly data for the past 5 years.
    """
    print("Reached Vector search 1")
    top_k = 10
    chunks = query_pinecone(query, top_k, year = year, quarter = quarter)
    contexts = "\n---\n".join(
        {chr(10).join([f'Chunk {i+1}: {chunk}' for i, chunk in enumerate(chunks)])}
    )
    return contexts

@tool("web_search")
def web_search(query: str):
    """Finds general knowledge STRICTLY about nvidia for the latest information using Google search.To add
    to your knowledgebase about the query. Focus stricitly only on Nvidia.
    """
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


def format_result(result):
    
    parts = []
    
    if 'data' in result and result['data']:
        parts.append("DATA:")
        for i, item in enumerate(result['data'], start=1):
            parts.append(f"Record {i}:")
            for key, value in item.items():
                parts.append(f"  {key.title()}: {value}")
            parts.append("")  
    
    if 'summary' in result:
        parts.append("SUMMARY:")
        summary = result['summary'].strip()
        parts.append(summary)
        parts.append("")
    
    if 'analysis_type' in result:
        parts.append("Analysis Type:")
        analysis_type = result['analysis_type'].strip()
        parts.append(analysis_type)
    return "\n".join(parts)


@tool("snowflake_query")
def snowflake_query(
    query: str = None, 
    analysis_type: str = 'financial_summary', 
    year: str = None, 
    quarter: List[str] = None
):
    """
    This tool fetches the required summaries and tables from structured NVIDIA data stored in snowflake
    queries table according to the query, year and quarters and returns, the table, data summary, analysis Type

    Strictly choose from the list analysis_type = ['financial_summary', 'stock_performance', 'balance_sheet', 'earnings_analysis']

    """
    result = snowflake_query_agent(
        query=query, 
        analysis_type=analysis_type, 
        year=year, 
        quarter=quarter
    )

    print("result 2:....->", format_result(result))
    
    return format_result(result)



@tool("final_answer")
def final_answer(
    research_steps: str,
    historical_performance: str,
    financial_analysis: str,
    analysis_type: str,
    industry_insights: str,
    summary: str,
    sources: str
):
    """
    Returns a comprehensive research report combining data from all agents.
    
    Args:
    -research_steps: Bullet points with each point explaining the step that wes taken
    to research your report. Format: Tool Name: step taken  (For each tool)
    - historical_performance: Detailed Analysis from RAG/vector_search agent. Write at least 2 paragraphs
    - financial_analysis: Write at least 2 paragraphs using previous response and Snowflake agent's financial summary and metrics. Also Display table in markdown format as well for the queries produced (for the columns that have data).If snowflake agent not used then a some sentence from other agents output
    - analysis_type: List of analysis types that was outputted from the snowflake tool -> under the Analysis Type heading of its output
    - industry_insights: Detailed Real-time trends from web search agent, Write at least 3 paragraphs
    - summary: A Fully detailed summary of at least 4 big paragraphs using all responses from all tools
    - sources: List of all referenced sources. Give links where possible
    
    Returns:
    Structured dictionary with complete research report components
    """
    if type(research_steps) is list:
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    if type(sources) is list:
        sources = "\n".join([f"- {s}" for s in sources])
    
    report = {
        "research_steps": research_steps if research_steps else "",
        "historical_performance": historical_performance,
        "financial_analysis": financial_analysis,
        "analysis_type": analysis_type,
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
    list of tools provided to you. If the query is completely irrelevant to NVIDIA, then inform them that It is not relevant.

    Context:
    - Year: {year or 'Not specified'}
    - Quarter: {quarter or 'Not specified'}

    Decice on the Tools usagae base usage based on the parameters
    Rules:
    - If a tool has been used with a particular query, do NOT use it again with the same query.
    - Do NOT use any tool more than **twice** (especially `web_search`). If a tool has already been used twice in the scratchpad, avoid using it again.
    - Collect information from a range of sources before providing a final answer.
    - Do not add bold or italics for any numerical value for the generated report

    If you see that a tool has been used (in the scratchpad) with a particular
    query, do NOT use that same tool with the same query again. Also, do NOT use
    any tool more than twice (ie, if the tool appears in the scratchpad twice, do
    not use it again).
    
    You should aim to collect information from a range of sources regarding NVIDIA before
    providing the answer to the user. Once you have collected the relevant information
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
    # return the tool name to use
    
    if isinstance(state["intermediate_steps"], list):
        return state["intermediate_steps"][-1].tool
    else:
        # if we output bad format go to final answer
        print("Router invalid format")
        return "final_answer"
    


def run_tool(state: AgentState):
    tool_str_to_func = {
            "web_search": web_search,
            "vector_search": vector_search,
            "snowflake_query": snowflake_query,
            "final_answer": final_answer
        }
    
    # use this as helper function so we repeat less code
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input

    if tool_name in ["vector_search"]:
        tool_args = {
            **tool_args,
            "year": state.get("year"),
            "quarter": state.get("quarter")
        }
    print(f"{tool_name}.invoke(input={tool_args})")
    # run tool
    out = tool_str_to_func[tool_name].invoke(input=tool_args)
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

    graph = StateGraph(AgentState)  # Keep type definition here

    # Pass state to all functions that require it
    graph.add_node("oracle", partial(run_oracle, oracle=research_agent))
    graph.add_node("web_search", run_tool)
    graph.add_node("vector_search", run_tool)
    graph.add_node("snowflake_query", run_tool)
    graph.add_node("final_answer", run_tool)

    graph.set_entry_point("oracle")

    graph.add_conditional_edges(
        source="oracle",
        path=router,
    )

    # create edges from each tool back to the oracle
    for tool_obj in tools:
        if tool_obj.name != "final_answer":
            graph.add_edge(tool_obj.name, "oracle")

    # if anything goes to final answer, it must then move to END
    graph.add_edge("final_answer", END)

    runnable = graph.compile()
    return runnable

def run_agents(tool_keys, year=None, quarter=None):
    research_agent = init_research_agent(tool_keys, year, quarter)
    runnable = create_graph(research_agent, year, quarter)
    return runnable
