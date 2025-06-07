from openai import OpenAI
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---- Step 1: Load and Chunk PDF ----
def load_pdf_chunks(pdf_path, chunk_size=500):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks

# ---- Step 2: Embed and Index the Chunks ----
def embed_chunks(chunks, embedder):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# ---- Step 3: Retrieve Relevant Chunks ----
def retrieve_context(query, chunks, embedder, index, embeddings, top_k=3):
    query_embedding = embedder.encode([query])
    _, indices = index.search(np.array(query_embedding), top_k)
    return "\n---\n".join([chunks[i] for i in indices[0]])


# ---- RAG Setup ----
pdf_path = "FAQ_for_Chatbot.pdf"  # <-- Replace with your actual PDF
chunks = load_pdf_chunks(pdf_path)
embedder = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
index, embeddings = embed_chunks(chunks, embedder)

# ---- User Query ----
user_input = input("Enter your question: ")
retrieved_context = retrieve_context(user_input, chunks, embedder, index, embeddings)

# ---- OpenAI API Call ----
client = OpenAI(
    api_key="sk-titCpkply6rFBcc6326yqTJzL20JeJJ9ACekZEij4nNCpClA",
    base_url="https://api.opentyphoon.ai/v1"
)

system_prompt = f"""
You are an AI assistant specialized in motherhood and childcare
Use the following additional context to help answer the question:
{retrieved_context}
"""

conversation = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_input}
]



# response = client.chat.completions.create(
#     model="typhoon-v2-70b-instruct",
#     messages=conversation,
#     stream=True,
#     max_tokens=512
# )

# for chunk in response:
#     if chunk.choices[0].delta.content is not None:
#         print(chunk.choices[0].delta.content, end="", flush=True)


print(chunks)
print("Retrieved Context:", retrieved_context)