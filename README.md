# DAMG7245_Assignment05_Part01

NvidiaFinRAG: Dynamic Multi-Agent is a dynamic pipeline that provides specific curated information based on the user’s specific queries regarding Nvidia’s financial dataset. It utilizes Langchain as its foundation and operates with three distinct agents to serve its knowledge base.
- Snowflake Agent - Serving its stocks and visual analysis 
- RAG Agent - Working to server from the extracted quarterly reports using embedding and vector database.
- Web Search - Provides the potential to fetch real time data analysis.

## Team Members

- Vedant Mane
- Abhinav Gangurde
- Yohan Markose

## Attestation:

WE ATTEST THAT WE HAVEN’T USED ANY OTHER STUDENTS’ WORK IN OUR ASSIGNMENT AND ABIDE BY THE POLICIES LISTED IN THE STUDENT HANDBOOK

## Resources

Application: [Streamlit Deployment](https://nvidia-finrag.streamlit.app/)

Backend API: [Google Cloud Run](https://fastapi-service-326204588719.us-east1.run.app)

Google Codelab: [Codelab](https://codelabs-preview.appspot.com/?file_id=1Lh26o3XUQmNmL8LVxyvQFOLbNgksNOY9QLW0UI6E08U/edit?tab=t.0#0)

Google Docs: [Project Document](https://docs.google.com/document/d/1Lh26o3XUQmNmL8LVxyvQFOLbNgksNOY9QLW0UI6E08U/edit?usp=sharing)

Video Walkthrough: [Video](https://northeastern-my.sharepoint.com/:v:/g/personal/markose_y_northeastern_edu/EQ1bM_uzSJ1Ks1CoizyLrqwB60K9hYq-H8zwND6WjjCBnA?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D&e=OZnleg)

## Technologies Used

- **Streamlit**: Frontend Framework
- **FastAPI**: API Framework
- **Google Cloud Run**: Backend Deployment
- **SerpAPI**: Website Data Extraction Open Source Tool
- **LangChain**: Implementation of RAG agents 
- **Snowflake**: Structured Data Strorage and Updation 
- **OpenAI**: Retrieval-Augmented Generation
- **Pinecone**: Vector Database

## Application Workflow Diagram

![Application Workflow](/architectural_diagram/Architectural_diagram.png)


### Workflow

1. **Data Ingestion**:
    - (Unstructured) Extracted reports (PDFs) are from the previous workflow are utilized for further processing.
    - For past 5 years worth of data has been processed and chunk are **converted into OpenAI embeddings** for semantic search.
        - The embeddings are stored in **Pinecone** for cloud-based vector search.
    - (Structured) Additional structured data has been has been scrapped and loaded from [Yahoo Finance](https://finance.yahoo.com/quote/NVDA/) into 4 Snowflake tables. 

2. **Query Processing & Retrieval**:
    - Users interact with a **Streamlit application**, where they can:
        - Select the **Year** and **Quarter** for the specific question.
        - Choose agent from the RAG system they want the question to be answered with options such as **Snowflake Agent** , **RAG Agent**  and **Web Search agent**.
        - Submit a **query** related to the nvidia financial aspect.
    - The query request is sent to **FastAPI**, which determines the retrieval method based on user selection.
    
3. **Agentic Multi-Agent System with LangGraph**
    - Using LangGraph, we have integrated and orchestrated the three specialized RAG agents to answer the query questions:
        - 1. Generic RAG Agent : Utilizing Pinecone and accessing filtered meta-data from stored Nvidia reports
        - 2. WebSearch Agent : Using SerpAPI and retriving information from top matching searches for any additional and latest context
        - 3. Snowflake Agent : Works on retrival of structured data scraped from Yahoo Finance for additional context from stocks preformace and visual representation 

4. **Response Generation & Display**:
    - Retrieved information are intergrated to passed to an **LLM** in this case OpenAI, which generates a relevant consolidated response.
    - The final output is sent back to the **Streamlit UI**, where users can view the extracted insights.

## **Data Flow & API Processes**

1. User Input
Users interact through the Streamlit UI, providing:
- Financial Queries: Users select the Year and Quarter for NVIDIA stock analysis.
- Agent Selection: Choose from Snowflake Agent, RAG Agent, or Web Search Agent for data retrieval.

2. Frontend (Streamlit UI)
The Streamlit application allows users to:
- Submit financial queries related to NVIDIA’s stock and reports.
- Select retrieval parameters (specific agent type).
- View AI-generated financial insights and stock visualizations.

3. Backend (FastAPI as API Gateway)
The FastAPI backend manages:
- Query Routing: Determines whether to use Snowflake, RAG, or Web Search.
- Data Processing: Retrieves structured/unstructured financial data.
- Response Generation: Integrates retrieved insights for AI-powered responses.

### **Google Cloud SDK Setup**
 
**Step 1: Download and Install Google Cloud SDK**
 
- Visit the [Google Cloud SDK documentation](https://cloud.google.com/sdk/docs/install) for platform-specific installation instructions.
- Download the installer for your operating system (Windows, macOS, or Linux).
- Follow the installation steps provided for your system.
 
**Step 2: Initialize Google Cloud SDK**
 
- Open a terminal or command prompt.
- Run `gcloud init` to begin the setup process.
- Follow the prompts to log in with your Google account and select a project.
 
**Step 3: Verify Installation**
 
- Run `gcloud --version` to confirm installation.
- Use `gcloud config list` to check the active configuration.
 

### Setting up the Docker Image on Google Cloud Run
 
1. **Build the Docker Image**
 
```docker
# Build and tag your image (make sure you're in the project directory)
docker build --platform=linux/amd64 --no-cache -t gcr.io/<YOUR_PROJECT_ID>/fastapi-app .
 
```
 
2. **Test Locally (Optional but Recommended)**
 
```docker
# Run the container locally
docker run -p 8080:8080 gcr.io/<YOUR_PROJECT_ID>/fastapi-app
 
# For Managing Environment Variables
docker run --env-file .env -p 8080:8080 gcr.io/<YOUR_PROJECT_ID>/fastapi-app
```
Visit http://localhost:8080/docs to verify the API works.
 
3. **Push to Google Container Registry**
 
```docker
# Push the image
docker push gcr.io/<YOUR_PROJECT_ID>/fastapi-app
```
 
4. **Deploy to Cloud Run**
 
```bash
gcloud run deploy fastapi-service \
  --image gcr.io/<YOUR_PROJECT_ID>/fastapi-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated  
```
 
5. **Get your Service URL**
 
```bash
gcloud run services describe fastapi-service \
  --platform managed \
  --region <REGION> \
  --format 'value(status.url)'
```
 
6. **Check Application Logs**
 
```bash
gcloud run services logs read fastapi-service --region <REGION>
```
 
## References

[Streamlit Documentation](https://docs.streamlit.io/) 

[FastAPI Documentation](https://fastapi.tiangolo.com/)

[Pinecone](https://www.pinecone.io/?utm_term=pinecone%20database&utm_campaign=brand-us-p&utm_source=adwords&utm_medium=ppc&hsa_acc=3111363649&hsa_cam=16223687665&hsa_grp=133738612775&hsa_ad=582256510975&hsa_src=g&hsa_tgt=kwd-1628011569744&hsa_kw=pinecone%20database&hsa_mt=p&hsa_net=adwords&hsa_ver=3&gad_source=1&gclid=CjwKCAjwnPS-BhBxEiwAZjMF0nFJVWpg9eEPcztz-TW5kQlc2pHrwV8O9KNX_jxqiIsfgm0-E3pUTBoCmxkQAvD_BwE)

[Snowflake Python APIs](https://docs.snowflake.com/en/developer-guide/snowflake-python-api/snowflake-python-overview)

[yfinance PYPI](https://pypi.org/project/yfinance/)

[Langchain](https://langchain-ai.github.io/langgraph/tutorials/introduction/Links)

Agents and workflows : [1](https://www.anthropic.com/engineering/building-effective-agents) [2](https://www.youtube.com/watch?v=usOmwLZNVuM) [3](https://weaviate.io/blog/what-is-agentic-rag)


