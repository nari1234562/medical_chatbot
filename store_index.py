from src.helper import (
    load_pdf_file,
    text_split,
    download_bedrock_embeddings
)

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY")

# Load PDF data
documents = load_pdf_file("Data/")

# Split text
text_chunks = text_split(documents)

# Bedrock embeddings
embeddings = download_bedrock_embeddings()

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

# Create index if not exists
if index_name not in [i.name for i in pc.list_indexes()]:

    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Store vectors
PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("✅ Vector DB created successfully")