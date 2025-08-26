import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI  # Corrected Import
from langchain.chains import RetrievalQA
import streamlit as st

load_dotenv()

MODEL_API_KEY = os.getenv("MODEL_API_KEY")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "health_collection")

def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=MODEL_API_KEY
    )
    db = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return db

def run():
    st.header("Chatbot")
    query = st.text_input("Ask a question about the ingested document:")
    if query:
        db = get_vectorstore()
        retriever = db.as_retriever()

        # Updated to use Google's LLM instead of OpenAI's
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", # Recommended for faster responses
            temperature=0,
            google_api_key=MODEL_API_KEY
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        answer = qa.run(query)
        st.write(answer)