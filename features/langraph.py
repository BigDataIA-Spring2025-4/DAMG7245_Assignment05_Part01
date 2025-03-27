from typing import TypedDict, Annotated, Optional, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator

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
from features.snowflake_data_pull import snowflake_pandaspull

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")



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
@tool("snowflake_agent")
def snowflake_agent(query: str):
    """
    This agent queries the Snowflake table FRED_DB.FRED_SCHEMA.NVDA_HISTORICAL_5Y.
    
    The table schema is:
      TICKER VARCHAR(10),
      DATE DATE,
      OPEN FLOAT,
      HIGH FLOAT,
      LOW FLOAT,
      CLOSE FLOAT,
      VOLUME FLOAT,
      DIVIDENDS FLOAT,
      STOCKSPLITS FLOAT

    The function passes the query parameter directly to snowflake_pandaspull.
    Provide the extact returned values from the snowflake_pandaspull dont elaborate or add any thing in the provided values.
    """

    contexts = snowflake_pandaspull(query)
    
    # Return the result
    return contexts



# Final Research Output Tool
@tool("final_answer")
def final_answer(
    introduction: str,
    research_steps: list,
    main_body: str,
    conclusion: str,
    snowflake_results : str,
    sources: list
):
    """Returns a natural language response to the user in the form of a research
    report. There are several sections to this report, those are:
    - `introduction`: a short paragraph introducing the user's question and the
    topic we are researching.
    - `research_steps`: a few bullet points explaining the steps that were taken
    to research your report.
    - `main_body`: this is where the bulk of high quality and concise
    information that answers the user's question belongs. It is 3-4 paragraphs
    long in length.
    - `conclusion`: this is a short single paragraph conclusion providing a
    concise but sophisticated view on what was found.
    - `snowflake_results` : If you have used snowflake_agent give the output you got from snowflake_pandaspull
    - `sources`: a bulletpoint list provided detailed sources for all information
    referenced during the research process
    """
    if type(research_steps) is list:
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    if type(sources) is list:
        sources = "\n".join([f"- {s}" for s in sources])
    return ""

def init_research_agent(tool_keys, year=None, quarter=None):
    tool_str_to_func = {
            "web_search": web_search,
            "vector_search": vector_search,
            "snowflake_agent": snowflake_agent,
            "final_answer": final_answer
        }
    
    tools = [final_answer]
    for val in tool_keys:
        tools.append(tool_str_to_func[val])

    ## Designing Agent Features and Prompt ##
    system_prompt = f"""You are the oracle, the great AI decision maker.
    Given the user's query you must decide what to do with it based on the
    list of tools provided to you.

    Context:
    - Year: {year or 'Not specified'}
    - Quarter: {quarter or 'Not specified'}
    - If you see the agents option "vector_search" selected the highest priority needs to provided to it for generating your final answer.
    - If year or quarter information is given, use that for query even for the "web_search" tool.

    If you see that a tool has been used (in the scratchpad) with a particular
    query, do NOT use that same tool with the same query again. Also, use all the tools atleast once but  do NOT use
    any tool more than twice (ie, if the tool appears in the scratchpad twice, do
    not use it again).

    You should aim to collect information from a diverse range of sources before
    providing the answer to the user. Once you have collected plenty of information
    to answer the user's question (stored in the scratchpad) use the final_answer tool."""

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
            "snowflake_agent": snowflake_agent,
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
        web_search,
        snowflake_agent,
        final_answer
    ]

    graph = StateGraph(AgentState)  # Keep type definition here

    # Pass state to all functions that require it
    graph.add_node("oracle", partial(run_oracle, oracle=research_agent))
    graph.add_node("web_search", run_tool)
    graph.add_node("snowflake_agent",run_tool)
    graph.add_node("vector_search", run_tool)
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
