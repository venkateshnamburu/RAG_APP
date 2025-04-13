import io
import re
import json
import fitz  # PyMuPDF
import numpy as np
from tqdm import tqdm
from azure.storage.blob import BlobServiceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import google.generativeai as genai
from pinecone import ServerlessSpec

from config import AZURE_CONNECTION_STRING, AZURE_CONTAINER_NAME, PINECONE_INDEX, pc


@st.cache_resource
def load_and_process_all_pdfs():
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    all_chunks = []
    for blob in container_client.list_blobs():
        if not blob.name.endswith(".pdf"):
            continue
        blob_client = container_client.get_blob_client(blob.name)
        pdf_data = blob_client.download_blob().readall()
        pdf = fitz.open(stream=io.BytesIO(pdf_data), filetype="pdf")
        first_page_text = pdf[0].get_text()
        lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
        candidate_name = lines[0] if lines else "Unknown"

        full_text = ""
        for page_num, page in enumerate(pdf, start=1):
            page_text = page.get_text().strip()
            full_text += f"\n\n--- Page {page_num} ---\n{page_text}"

        chunks = splitter.create_documents(
            texts=[full_text],
            metadatas=[{"filename": blob.name, "candidate_name": candidate_name}]
        )
        all_chunks.extend(chunks)
    return all_chunks


@st.cache_resource
def embed_chunks(_all_chunks):
    embedded = []
    for i, chunk in enumerate(_all_chunks):
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=chunk.page_content,
                task_type="retrieval_document"
            )
            embedding_vector = response["embedding"]
            embedded.append({
                "id": f"{chunk.metadata['filename']}_{i}",
                "text": chunk.page_content,
                "embedding": embedding_vector,
                "metadata": chunk.metadata
            })
        except Exception as e:
            st.warning(f"Embedding error: {e}")
    return embedded


@st.cache_resource
def initialize_index():
    # Hardcoding cloud and region values
    cloud = "aws"  # or "gcp" depending on your Pinecone region
    region = "us-west1"  # Set your Pinecone region here (e.g., us-west1, eu-central1, etc.)

    # Check if the index already exists and delete it if it does
    if PINECONE_INDEX in [idx.name for idx in pc.list_indexes()]:
        pc.delete_index(PINECONE_INDEX)
        
    # Create the Pinecone index with the correct cloud and region values
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region)
    )
    
    return pc.Index(PINECONE_INDEX)


@st.cache_resource
def upsert_to_pinecone(embedded_chunks):
    index = initialize_index()
    for i in tqdm(range(0, len(embedded_chunks), 100)):
        batch = embedded_chunks[i:i + 100]
        vectors = [
            {
                "id": chunk["id"],
                "values": chunk["embedding"],
                "metadata": {"text": chunk["text"], **chunk["metadata"]}
            }
            for chunk in batch
        ]
        index.upsert(vectors=vectors)
    return index


def calculate_metrics(retrieved_texts, correct_answer, model_answer):
    answer_accuracy = 1 if model_answer.lower() == correct_answer.lower() else 0
    relevant = [1 if correct_answer.lower() in text.lower() else 0 for text in retrieved_texts]
    precision_at_k = sum(relevant) / len(retrieved_texts) if retrieved_texts else 0
    recall_at_k = 1.0 if any(relevant) else 0
    mrr = 0
    for idx, relevant_flag in enumerate(relevant):
        if relevant_flag:
            mrr = 1 / (idx + 1)
            break
    return {
        "answer_accuracy": answer_accuracy,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr
    }


def extract_json(text):
    try:
        json_text = re.search(r"\{.*\}", text, re.DOTALL).group()
        return json.loads(json_text)
    except Exception:
        return None
