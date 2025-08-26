import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
# You'll need to install this: pip install langchain-text-splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

MODEL_API_KEY = os.getenv("MODEL_API_KEY")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "health_collection")

def ingest_file(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Create a text splitter for better chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Split the documents into smaller chunks
    splits = text_splitter.split_documents(docs)

    # Use Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=MODEL_API_KEY
    )

    # Create Chroma vectorstore from the Document objects
    vectorstore = Chroma.from_documents(
        documents=splits,  # Pass the list of Document objects here
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory="./chroma_db"
    )
    vectorstore.persist()
    st.success("Document ingested and embeddings stored!")

def run():
    st.header("Ingest PDF Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        ingest_file("temp.pdf")