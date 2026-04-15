from langchain_community.vectorstores import FAISS
from src.helper import file_loader, chunking_data, get_embedding
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from src.helper import get_llm 
import os

VECTOR_PATH = "vector_store"

def build_vector_store():
    docs = file_loader("data")
    chunks = chunking_data(docs)

    embeddings = get_embedding()

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_PATH)

    return db


def load_vector_store():
    embeddings = get_embedding()

    return FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_retriever(k=3):
    db = load_vector_store()

    base_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 10
        }
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=get_llm()
    )

    return retriever