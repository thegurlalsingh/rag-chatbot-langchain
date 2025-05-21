import os
import google.generativeai as genai
from PyPDF2 import PdfReader
import docx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

GEMINI_API_KEY = "AIzaSyBfMK8WdtVLTFNtzrz3GLuTcl4lnFxmqAQ"  # <-- Your Gemini API key here
genai.configure(api_key=GEMINI_API_KEY)

# Loaders
def load_txt(file): return file.read().decode('utf-8')

def load_pdf(file):
    pdf = PdfReader(file)
    return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def load_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_file(file):
    ext = os.path.splitext(file.name)[1]
    if ext == ".pdf": return load_pdf(file)
    elif ext == ".docx": return load_docx(file)
    elif ext == ".txt": return load_txt(file)
    else: return ""

# Chunking
def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Embedding and FAISS
def build_vectorstore(text_chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, text_chunks, model

# QA with Gemini
def query_with_gemini(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()
