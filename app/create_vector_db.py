import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Paths
INPUT_FILE = "data/cleaned_chunks.jsonl"
VECTOR_DB_PATH = "data/guvi_faiss.index"
META_FILE = "data/guvi_metadata.json"

# Step 1: Load preprocessed chunks
print("📥 Loading cleaned data...")
texts, metadata = [], []
with open(INPUT_FILE, "r", encoding="utf8") as f:
    for line in f:
        item = json.loads(line)
        texts.append(item["text"])
        metadata.append({"id": item["id"], "source": item["source"]})

print(f"✅ Loaded {len(texts)} text chunks.")

# Step 2: Load embedding model
print("🔍 Loading embedding model (this may take a minute)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Generate embeddings
print("⚙️ Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Step 4: Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, VECTOR_DB_PATH)

# Step 5: Save metadata
with open(META_FILE, "w", encoding="utf8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("✅ FAISS vector database created successfully!")
print(f"🧠 Index saved to: {VECTOR_DB_PATH}")
print(f"🗂 Metadata saved to: {META_FILE}")
