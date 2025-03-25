import openai, os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from chunk_strategy import semantic_chunking
load_dotenv()

from services.s3 import S3FileManager

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def connect_to_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if not pc.has_index(PINECONE_INDEX):
        pc.create_index(
            name=PINECONE_INDEX,
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
    index = pc.Index(PINECONE_INDEX)
    return index

def read_markdown_file(file, s3_obj):
    content = s3_obj.load_s3_file_content(file)
    return content

def get_embedding(text):
    """Generates an embedding for the given text using OpenAI."""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def create_pinecone_vector_store(file, chunks):
    index = connect_to_pinecone_index()
    vectors = []
    file = file.split('/')
    # parser = file[1]
    # identifier = file[2]
    year = file[1]
    quarter = file[2]
    records = 0
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        vectors.append((
            f"id_{year}_{quarter}_chunk_{i}",  # Unique ID
            embedding,  # Embedding vector
            {"year": year, "quarter": quarter, "text": chunk}  # Metadata
        ))
        if len(vectors) >= 20:
            records += len(vectors)
            upsert_vectors(index, vectors)
            # index.upsert(vectors=vectors, namespace=f"{parser}_{chunk_strategy}")
            print(f"Inserted {len(vectors)} chunks into Pinecone.")
            vectors.clear()
    # Store in Pinecone under the correct namespace
    if len(vectors)>0:
        upsert_vectors(index, vectors)
        # index.upsert(vectors=vectors, namespace=f"{parser}_{chunk_strategy}")
        print(f"Inserted {len(vectors)} chunks into Pinecone.")
        records += len(vectors)
    print(f"Inserted {records} chunks into Pinecone.")

def upsert_vectors(index, vectors):
    index.upsert(vectors=vectors, namespace=f"nvdia_quarterly_reports")

def query_pinecone(parser, chunking_strategy, query, top_k=20, year = None, quarter = None):
    # Search the dense index and rerank the results
    index = connect_to_pinecone_index()
    dense_vector = get_embedding(query)
    results = index.query(
        namespace=f"{parser}_{chunking_strategy}",
        vector=dense_vector,  # Dense vector embedding
        filter={
            "year": {"$eq": year},
            "quarter": {"$in": quarter},
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


def main():
    print("Hello, World!")


if __name__ == "__main__":
    # Load the OpenAI API key from the environment
    base_path = "nvidia"
    s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)
    # files = list({file for file in s3_obj.list_files() if file.endswith('.md')})
    
    year = ['2021', '2022', '2023', '2024', '2025']
    quarter = ['Q1', 'Q2', 'Q3', 'Q4']
    
    for y in year:
        for q in quarter:
            file = f"{base_path}/{y}/{q}/mistral/extracted_data.md"
            print(f"Reading file for {y}-{q}")
            content = read_markdown_file(file, s3_obj)
            if len(content) != 0:
                print(f"Successfully read file for {y}-{q}")
                print("Implementing semantic chunking")
                chunks = semantic_chunking(content, max_sentences=10)
                if len(chunks) != 0:
                    print("Successfully chunked the content")
                    print("Creating Pinecone vector store")
                    create_pinecone_vector_store(file, chunks)
                # create_pinecone_vector_store(file, chunks, "semantic")
    # for file in files:
    #     print(file)
    # print(len(files))

    main()
    