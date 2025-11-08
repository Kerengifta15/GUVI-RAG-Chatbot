import streamlit as st
import google.generativeai as genai
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import time
from datetime import datetime

# STREAMLIT PAGE CONFIGURATION

st.set_page_config(page_title="GUVI RAG Chatbot", page_icon="🤖", layout="wide")

# CONFIGURE GEMINI API

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found. Please set it using:\n`setx GEMINI_API_KEY your_api_key_here` in CMD.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel("models/gemini-2.5-flash")

# LOAD DATASET

data_files = [
    "data/www_guvi_in_faq.txt",
    "data/guvi_faq.txt"
]
texts = []
for file in data_files:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            texts.append(f.read())
combined_text = " ".join(texts)
st.write(f"✅ Loaded {len(combined_text.split())} words from GUVI data.")

# TEXT CHUNKING

def chunk_text(text, chunk_size=800):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = chunk_text(combined_text)

# EMBEDDING MODEL + FAISS INDEX (CACHED)

@st.cache_resource
def load_embedder_and_index():
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return embedder, index

embedder, index = load_embedder_and_index()

# SIDEBAR

with st.sidebar:
    st.title("⚙️ App Controls")
    st.markdown("**Developed by:** Keren Gifta 🌸")
    st.markdown("---")

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

    if "chat_history" in st.session_state and st.session_state.chat_history:
        chat_text = ""
        for q, a in st.session_state.chat_history:
            chat_text += f"User: {q}\nBot: {a}\n\n"
        st.download_button(
            label="⬇️ Download Conversation",
            data=chat_text,
            file_name=f"GUVI_Chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# MAIN INTERFACE

st.title("🤖 GUVI RAG Chatbot")
st.markdown("### Ask any question about GUVI (FAQ, About, Blog info)!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("💬 Type your question below:")

if st.button("Ask") and user_query.strip():
    # Start query process (removed search info message)
    time.sleep(0.3)

    query_embedding = embedder.encode([user_query], convert_to_tensor=False)
    D, I = index.search(np.array(query_embedding, dtype="float32"), k=3)
    retrieved_chunks = [chunks[i] for i in I[0]]
    st.write(f"✅ Retrieved top {len(I[0])} chunks for your query.")

    chat_context = ""
    for q, a in st.session_state.chat_history[-3:]:
        chat_context += f"User: {q}\nAssistant: {a}\n"

    context = "\n".join(retrieved_chunks)
    prompt = f"""
You are a helpful chatbot that answers based on GUVI's website data.

Conversation history:
{chat_context}

Context:
{context}

User question: {user_query}

Answer concisely and clearly.
"""

    try:
        with st.spinner("🔎 Searching relevant GUVI data... Please wait..."):
            response = model_gemini.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"⚠️ Gemini Error: {e}"

    st.session_state.chat_history.append((user_query, answer))

# CHAT DISPLAY

if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Conversation")

    for i, (q, a) in enumerate(st.session_state.chat_history[-10:], 1):
        st.markdown(
            f"""
            <div style='background-color:#DCF8C6; padding:10px; border-radius:10px; margin-bottom:8px;'>
                <b>🧑 You:</b> {q}
            </div>
            <div style='background-color:#F1F0F0; padding:10px; border-radius:10px; margin-bottom:20px;'>
                <b>🤖 Bot:</b> {a}
            </div>
            """,
            unsafe_allow_html=True
        )
