import os  # For file path operations
import streamlit as st  # Web framework for creating UI
import faiss  # Library for fast similarity search
import numpy as np  # Numerical operations, mainly for vector normalization
import json  # To read/write JSON files
from transformers import pipeline  # NLP pipeline for text generation
from rank_bm25 import BM25Okapi  # BM25 retrieval for document ranking
from sentence_transformers import SentenceTransformer  # Embedding model for vector retrieval

# Define data directory and file paths
DATA_DIR = "data"  # Relative path to store necessary files
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_pdfs.bin")  # FAISS index file for vector search
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "pdf_chunks.json")  # JSON file containing PDF chunks metadata
TABLES_JSON_PATH = os.path.join(DATA_DIR, "financial_tables.json")  # JSON file for financial table data
DOCS_JSON_PATH = os.path.join(DATA_DIR, "all_docs.json")  # JSON file storing all document texts

# Ensure the FAISS index file exists before proceeding
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"FAISS index file {FAISS_INDEX_PATH} not found! Run embedder.py first.")

# Load FAISS index into memory
faiss_pdf = faiss.read_index(FAISS_INDEX_PATH)
print(f"FAISS index loaded with {faiss_pdf.ntotal} entries.")

# Load chunk metadata containing text snippets of PDF chunks
with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
    chunk_metadata = json.load(f)

# Load financial table data if available, otherwise initialize as empty list
if os.path.exists(TABLES_JSON_PATH):
    with open(TABLES_JSON_PATH, "r", encoding="utf-8") as f:
        financial_tables = json.load(f)
else:
    financial_tables = []

# Initialize SentenceTransformer model for text embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_similar_documents(query, top_k=3):
    """Performs single-stage retrieval using FAISS index and structured financial data."""
    # Encode query into an embedding vector and normalize
    query_emb = model.encode([query], convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb)

    # Perform nearest neighbor search on FAISS index
    distances, indices = faiss_pdf.search(query_emb, top_k)
    results = {"PDF Results": [], "Structured Financial Data": []}

    # Retrieve the top-k PDF chunks based on FAISS search results
    for i in range(top_k):
        idx = indices[0][i]  # Chunk index
        dist = distances[0][i]  # Distance score
        chunk_info = chunk_metadata[idx]
        snippet = chunk_info["text"]
        pdf_file_name = chunk_info["pdf_file"]
        results["PDF Results"].append(f"[{pdf_file_name}] chunk #{idx}, distance={dist:.4f}\n{snippet}")

    # Search for relevant financial data in structured tables
    structured_hits = []
    for entry in financial_tables:
        if "data" in entry:
            for row in entry["data"]:
                row_str = " | ".join(str(x) for x in row.values())
                if query.lower() in row_str.lower():
                    structured_hits.append(row)

    results["Structured Financial Data"] = structured_hits if structured_hits else ["No structured data found."]
    return results

# Ensure that document-level JSON exists for BM25 retrieval
if not os.path.exists(DOCS_JSON_PATH):
    raise FileNotFoundError("Need doc-level JSON for BM25 coarse retrieval!")

# Load all document texts for BM25 retrieval
with open(DOCS_JSON_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)

# Initialize BM25 retriever with tokenized documents
bm25_corpus = [doc["text"].split() for doc in documents]
bm25 = BM25Okapi(bm25_corpus)

def multi_stage_retrieve(query, top_k_coarse=3, top_k_fine=3):
    """Performs multi-stage retrieval using BM25 and FAISS."""
    # Stage 1: Coarse retrieval using BM25
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    doc_ranking = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    top_doc_indices = doc_ranking[:top_k_coarse]

    # Retrieve chunks from top-ranked documents
    candidate_chunk_indices = []
    for i in top_doc_indices:
        doc_id = documents[i].get("doc_id", None)
        if doc_id is None:
            continue
        for idx, meta in enumerate(chunk_metadata):
            if meta.get("doc_id") == doc_id:
                candidate_chunk_indices.append(idx)

    # Stage 2: Fine retrieval using FAISS
    query_emb = model.encode([query], convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb)
    distances, indices = faiss_pdf.search(query_emb, top_k_fine * 10)

    # Filter relevant chunks
    all_hits = [(idx, dist) for dist, idx in zip(distances[0], indices[0]) if idx in candidate_chunk_indices]
    all_hits.sort(key=lambda x: x[1])
    final_hits = all_hits[:top_k_fine]

    # Prepare results
    results = [{"chunk_id": idx, "distance": dist, "text": chunk_metadata[idx]["text"], "pdf_file": chunk_metadata[idx]["pdf_file"]} for idx, dist in final_hits]
    return results

# Initialize Flan-T5 text generation model pipeline
model_name = "google/flan-t5-small"
generator_pipeline = pipeline("text2text-generation", model=model_name)

def generate_response(query, mode="basic"):
    """Generates response by retrieving relevant documents and using Flan-T5 for answer generation."""
    # Retrieve top chunks based on selected retrieval mode
    if mode == "multi-stage":
        retrieved_hits = multi_stage_retrieve(query)
        top_chunks = [f"[{r['pdf_file']}] chunk #{r['chunk_id']}, distance={r['distance']:.4f}\n{r['text']}" for r in retrieved_hits]
        structured_data = ["No structured data for multi-stage."]
    else:
        results = retrieve_similar_documents(query)
        top_chunks = results["PDF Results"]
        structured_data = results["Structured Financial Data"]

    # Construct input prompt for Flan-T5 model
    prompt_intro = "You are a financial Q&A assistant. Use the data below.\n\n"
    context_chunks = "\n\n---\n\n".join(top_chunks)
    structured_text = "\n".join(str(row) for row in structured_data)
    final_prompt = f"{prompt_intro}Query: {query}\n\nRelevant PDF Chunks:\n{context_chunks}\n\nStructured Data:\n{structured_text}\n\nProvide a concise, accurate answer:\n"

    # Call text generation pipeline
    output = generator_pipeline(final_prompt, max_length=256)
    return output[0]["generated_text"]

def main():
    """Streamlit app interface."""
    st.title("RAG Financial Q&A")
    retrieval_mode = st.selectbox("Select Retrieval Mode:", ["basic", "multi-stage"])
    user_query = st.text_input("Enter your query here:")
    if st.button("Submit") and user_query.strip():
        with st.spinner("Generating answer..."):
            answer = generate_response(user_query, mode=retrieval_mode)
            st.markdown("### Answer")
            st.write(answer)

if __name__ == "__main__":
    main()
