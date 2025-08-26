import nest_asyncio
nest_asyncio.apply()
import os
from pages import ingest_page, chatbot_page
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot System")

menu = ["Ingest Document", "Chatbot"]
choice = st.sidebar.selectbox("Select Option", menu)

if choice == "Ingest Document":
    ingest_page.run()
elif choice == "Chatbot":
    chatbot_page.run()