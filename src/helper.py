from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_aws import BedrockEmbeddings


def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


def text_split(extracted_data):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    return text_splitter.split_documents(extracted_data)


def download_bedrock_embeddings():

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name="us-east-1"
    )

    return embeddings