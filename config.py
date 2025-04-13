# config.py
import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

# ---- Required Secret Keys ----
required_keys = [
    "AZURE_CONNECTION_STRING",
    "AZURE_CONTAINER_NAME",
    "GOOGLE_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_ENVIRONMENT",
    "PINECONE_INDEX"
]

# ---- Validate secrets ----
missing_keys = [key for key in required_keys if key not in st.secrets]
if missing_keys:
    st.error(f"❌ Missing keys in `.streamlit/secrets.toml`: {', '.join(missing_keys)}")
    st.stop()

# ---- Load secrets ----
AZURE_CONNECTION_STRING = st.secrets["AZURE_CONNECTION_STRING"]
AZURE_CONTAINER_NAME = st.secrets["AZURE_CONTAINER_NAME"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]
PINECONE_INDEX = st.secrets["PINECONE_INDEX"]

# ---- Configure APIs ----
genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
