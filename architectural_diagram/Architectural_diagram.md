flowchart TB
    subgraph User["User Interface"]
        UI[Streamlit App]
    end
    
    subgraph API["API Layer"]
        FastAPI[FastAPI Backend]
    end
    
    subgraph LangGraph["LangGraph Orchestration"]
        Oracle[Research Oracle/Agent]
        Router{Router}
        
        subgraph Agents["Specialized Agents"]
            RAG[RAG Agent]
            Snowflake[Snowflake Agent]
            WebSearch[Web Search Agent]
        end
        
        FinalAnswer[Final Answer Tool]
    end
    
    subgraph DataSources["Data Sources"]
        PineconeDB[(Pinecone Vector DB)]
        SnowflakeDB[(Snowflake DB)]
        SerpAPI[SerpAPI/Google]
    end
    
    UI --> FastAPI
    FastAPI --> Oracle
    
    Oracle --> Router
    Router --> RAG
    Router --> Snowflake
    Router --> WebSearch
    
    RAG --> Oracle
    Snowflake --> Oracle
    WebSearch --> Oracle
    
    Oracle --> FinalAnswer
    FinalAnswer --> FastAPI
    FastAPI --> UI
    
    RAG <--> PineconeDB
    Snowflake <--> SnowflakeDB
    WebSearch <--> SerpAPI
  
