import os, time, base64, asyncio, chromadb, tempfile
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from services.s3 import S3FileManager
from features.chunking.chunk_strategy import markdown_chunking, semantic_chunking, sliding_window_chunking
from features.pinecone.pinecone_openai import connect_to_pinecone_index, get_embedding, query_pinecone
from features.chromadb.chromadb_openai import get_chroma_embeddings, query_chromadb
from features.manual_rag.manual_stores import generate_response_manual, create_manual_vector_store_doc

from openai import OpenAI
from features.pdf_extraction.docling_pdf_extractor import pdf_docling_converter
from features.pdf_extraction.mistralocr_pdf_extractor import pdf_mistralocr_converter
from pinecone import Pinecone, ServerlessSpec

class PdfInput(BaseModel):
    file: str
    file_name: str
    parser: str

class NVDIARequest(BaseModel):
    year: str
    quarter: list
    parser: str
    chunk_strategy: str
    vector_store: str
    query: str

class DocumentQueryRequest(BaseModel):
    parser: str
    chunk_strategy: str
    vector_store: str
    file_name: str
    markdown_content: str
    query: str
    
app = FastAPI()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_FILE_INDEX = os.getenv("PINECONE_FILE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

client = OpenAI()

@app.get("/")
def read_root():
    return {"message": "NVDIA Financial Reports Analysis: FastAPI Backend with OpenAI Integration available for user queries..."}

@app.post("/upload_pdf")
def process_pdf_docling(uploaded_pdf: PdfInput):
    print("Convertiung files")
    pdf_content = base64.b64decode(uploaded_pdf.file)
    # Convert pdf_content to a BytesIO stream for pymupdf
    pdf_stream = BytesIO(pdf_content)
    # base_path = f"pdf/docling/{uploaded_pdf.file_name.replace('.','').replace(' ','')}_{timestamp}/"
    base_path = f"pdf/{uploaded_pdf.parser}/{uploaded_pdf.file_name.replace('.','').replace(' ','')}"
    s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)
    s3_obj.upload_file(AWS_BUCKET_NAME, f"{s3_obj.base_path}/{uploaded_pdf.file_name}", pdf_content)
    if uploaded_pdf.parser == "docling":
        file_name, result = pdf_docling_converter(pdf_stream, base_path, s3_obj)
    else:
        file_name, result = pdf_mistralocr_converter(pdf_stream, base_path, s3_obj)        
    return {
        "message": f"Data Scraped and stored in S3 \n Click the link to Download: https://{s3_obj.bucket_name}.s3.amazonaws.com/{file_name}",
        "file_name": file_name,
        "scraped_content": result  # Include the original scraped content in the response
    }

@app.post("/query_document")
async def query_document(request: DocumentQueryRequest):
    try:
        file_name = request.file_name
        markdown_content = request.markdown_content
        query = request.query
        parser = request.parser
        chunk_strategy = request.chunk_strategy
        vector_store = request.vector_store
        top_k = 10
        chunks = generate_chunks(markdown_content, chunk_strategy)
        print(f"Chunks size: {len(chunks)}")
        answer = ""
        if vector_store == "pinecone":
            await create_pinecone_vector_store(file_name, chunks, chunk_strategy, parser)
            result_chunks = generate_response_doc_pinecone(file = file_name, parser = parser, chunking_strategy = chunk_strategy, query = query, top_k=top_k)
            if len(result_chunks) == 0:
                raise HTTPException(status_code=500, detail="No relevant data found in the document")
            else:
                # message = generate_openai_message(chunks, year, quarter, query)
                # answer = generate_model_response(message)
                message = generate_openai_message_document(query, result_chunks)
                answer = generate_model_response(message)
        elif vector_store == "chromadb":
            s3_obj = await create_chromadb_vector_store(file_name, chunks, chunk_strategy, parser)
            result_chunks = generate_response_doc_chroma(file_name, parser, chunk_strategy, query, top_k, s3_obj)
            # query_chromadb_doc(file_name, parser, chunk_strategy, query, top_k, s3_obj)
            message = generate_openai_message_document(query, result_chunks)
            print(message)
            answer = generate_model_response(message)
            print(answer)
        
        elif vector_store == "manual":
            s3_obj = create_manual_vector_store_doc(file_name, chunks, chunk_strategy, parser)
            result_chunks = generate_response_manual(s3_obj, parser, chunk_strategy, query, top_k)
            
            message = generate_openai_message_document(query, result_chunks)
            print(message)
            answer = generate_model_response(message)
            print(answer)

        return {
            # "answer": parser + chunk_strategy + vector_store + query + file_name + str(len(chunks)),
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.post("/query_research_agent")
def query_nvdia_documents(request: NVDIARequest):
    try:
        year = request.year
        quarter = request.quarter
        query = request.query
        tools = request.tools

        print("Tools selected are:", tools)
        answer = ''
        
        return {
            # "answer": year + quarter[0] + parser + chunk_strategy + vector_store + query,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

def generate_chunks(markdown_content, chunk_strategy):
    chunks = list()
    if chunk_strategy == "markdown":
        chunks = markdown_chunking(markdown_content, heading_level=2)
    elif chunk_strategy == "semantic":
        chunks = semantic_chunking(markdown_content, max_sentences=10)
    elif chunk_strategy == "sliding_window":
        chunks = sliding_window_chunking(markdown_content, chunk_size=1000, overlap=150)
    else:
        pass
    return chunks
    
    
def generate_response_from_pinecone(parser, chunk_strategy, query, top_k, year, quarter ):
    response = query_pinecone(parser = parser, chunking_strategy = chunk_strategy, query = query, top_k=top_k, year = year, quarter = quarter)
    return response

def generate_response_doc_pinecone(file, parser, chunking_strategy, query, top_k=10):
    response = query_pinecone_doc(file, parser, chunking_strategy, query, top_k=top_k)
    return response

def generate_response_from_chroma(parser, chunk_strategy, query, top_k, year, quarter ):
    response = query_chromadb(parser = parser, chunking_strategy = chunk_strategy, query = query, top_k=top_k, year = year, quarter = quarter)
    return response

def generate_response_doc_chroma(file_name, parser, chunking_strategy, query, top_k, s3_obj):
    response = query_chromadb_doc(file_name, parser, chunking_strategy, query, top_k, s3_obj)
    return response




def generate_openai_message_document(query, chunks):
    prompt = f"""
    Below are relevant excerpts from a document uploaded by the yser that may help answer the user query.

    --- User Query ---
    {query}

    --- Relevant Document Chunks ---
    {chr(10).join([f'Chunk {i+1}: {chunk}' for i, chunk in enumerate(chunks)])}

    Based on the provided document chunks, generate a comprehensive response to the query. If needed, synthesize the information and ensure clarity.
    """
    # print(prompt)
    return prompt
    
def generate_openai_message(chunks, year, quarter, query):
    prompt = f"""
    Below are relevant excerpts from a NVDIA quarterly financial report for year {year} and quarter {quarter} that may help answer the query.

    --- User Query ---
    {query}

    --- Relevant Document Chunks ---
    {chr(10).join([f'Chunk {i+1}: {chunk}' for i, chunk in enumerate(chunks)])}

    Based on the provided document chunks, generate a comprehensive response to the query. If needed, synthesize the information and ensure clarity.
    """
    print(prompt)
    return prompt

def connect_to_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if not pc.has_index(PINECONE_FILE_INDEX):
        pc.create_index(
            name=PINECONE_FILE_INDEX,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
            tags={
                "environment": "development"
            }
        )
    index = pc.Index(PINECONE_FILE_INDEX)
    return index

async def create_pinecone_vector_store(file, chunks, chunk_strategy, parser):
    index = connect_to_pinecone_index()
    namespace = f"{file}_{parser}_{chunk_strategy}"
    try:
        index.delete(delete_all=True, namespace=namespace)
    except:
        pass
    vectors = []
    records = 0
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        vectors.append((
            f"{file}_chunk_{i}",  # Unique ID
            embedding,  # Embedding vector
            {"file": file, "text": chunk}  # Metadata
        ))
        if len(vectors) >= 20:
            records += len(vectors)
            upsert_vectors(index, vectors, namespace)
            # index.upsert(vectors=vectors, namespace=f"{parser}_{chunk_strategy}")
            print(f"Inserted {len(vectors)} chunks into Pinecone.")
            vectors.clear()
    # Store in Pinecone under the correct namespace
    if len(vectors)>0:
        upsert_vectors(index, vectors, namespace)
        # index.upsert(vectors=vectors, namespace=f"{parser}_{chunk_strategy}")
        print(f"Inserted {len(vectors)} chunks into Pinecone.")
        records += len(vectors)
    print(f"Inserted {records} chunks into Pinecone.")

def upsert_vectors(index, vectors, namespace):
    index.upsert(vectors=vectors, namespace=namespace)

def query_pinecone_doc(file, parser, chunking_strategy, query, top_k=10):
    # Search the dense index and rerank the results
    index = connect_to_pinecone_index()
    dense_vector = get_embedding(query)
    # print(dense_vector)
    print(file)
    namespace = f"{file}_{parser}_{chunking_strategy}"
    print(namespace)
    results = index.query(
        namespace=namespace,
        vector=dense_vector,  # Dense vector embedding
        filter={
            "file": {"$eq": file},
        },  # Sparse keyword match
        top_k=top_k,
        include_metadata=True,  # Include chunk text
    )
    responses = []
    for match in results["matches"]:
        print(f"ID: {match['id']}, Score: {match['score']}")
        # print(f"Chunk: {match['metadata']['text']}\n")
        responses.append(match['metadata']['text'])
        print("=================================================================================")
    return responses

async def create_chromadb_vector_store(file, chunks, chunk_strategy, parser):
    with tempfile.TemporaryDirectory() as temp_dir:
        chroma_client = chromadb.PersistentClient(path=temp_dir)
        print(file)
        file_name = file.split('/')[2]
        print(file_name)
        base_path = "/".join(file.split('/')[:-1])
        print(base_path)
        s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)
        # create_chromadb_vector_store(chroma_client, file, chunks_mark, chunk_strategy)
        collection_file = chroma_client.get_or_create_collection(name=f"{file_name}_{parser}_{chunk_strategy}")
        base_metadata = {
            "file": file_name
        }
        metadata = [base_metadata for _ in range(len(chunks))]
        
        embeddings = get_chroma_embeddings(chunks)
        ids = [f"{file_name}_{parser}_{chunk_strategy}_{i}" for i in range(len(chunks))]
        
        collection_file.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata,
            documents=chunks
        )
        # Upload the entire ChromaDB directory to S3
        upload_directory_to_s3(temp_dir, s3_obj, "chroma_db")
    print("ChromaDB has been uploaded to S3.")
    return s3_obj

def upload_directory_to_s3(local_dir, s3_obj, s3_prefix):
    """Upload a directory and its contents to S3"""
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # Create the S3 key by replacing the local directory path with the S3 prefix
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = f"{s3_obj.base_path}/{os.path.join(s3_prefix, relative_path)}".replace("\\", "/")
            
            with open(local_path, "rb") as f:
                s3_obj.upload_file(AWS_BUCKET_NAME, s3_key, f.read())

def download_chromadb_from_s3(s3_obj, temp_dir):
    """Download ChromaDB files from S3 to a temporary directory"""
    s3_prefix = f"{s3_obj.base_path}/chroma_db"
    s3_files = [f for f in s3_obj.list_files() if f.startswith(s3_prefix)]
    
    for s3_file in s3_files:
        # Extract the relative path from the S3 key
        relative_path = s3_file[len(s3_prefix):].lstrip('/')
        local_path = os.path.join(temp_dir, relative_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download the file from S3
        content = s3_obj.load_s3_pdf(s3_file)
        with open(local_path, 'wb') as f:
            f.write(content if isinstance(content, bytes) else content.encode('utf-8'))

def query_chromadb_doc(file_name, parser, chunking_strategy, query, top_k, s3_obj):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # s3_obj = S3FileManager(AWS_BUCKET_NAME, "")
            download_chromadb_from_s3(s3_obj, temp_dir)
            chroma_client = chromadb.PersistentClient(path=temp_dir)
            file_name = file_name.split('/')[2]

            try:
                collection = chroma_client.get_collection(f"{file_name}_{parser}_{chunking_strategy}")
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")
            
            # Create embeddings for the query
            query_embeddings = get_chroma_embeddings([query])
            
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k
            )
            
            return results["documents"]
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error querying ChromaDB: {str(e)}")

def generate_model_response(message):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. You are given excerpts from NVDIA's quarterly financial report. Use them to answer the user query."},
                {"role": "user", "content": message}
            ],
            max_completion_tokens=2048
        )
        # print(response.choices[0].message)
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response from OpenAI Model: {str(e)}")
    
    