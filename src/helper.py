from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_openai import ChatOpenAI
import os

def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )


def file_loader(path):
    loader = DirectoryLoader(
        path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def chunking_data(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,   
        chunk_overlap=100 
    )
    chunks = splitter.split_documents(data)
    return chunks


def get_embedding():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small" 
    )
    return embeddings