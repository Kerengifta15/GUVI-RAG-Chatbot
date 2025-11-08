import os
import re
import json
from tqdm import tqdm

DATA_DIR = "data"
OUTPUT_FILE = "data/cleaned_chunks.jsonl"

def clean_text(text):
    """Remove unwanted characters and extra spaces."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\xa0', ' ')
    text = re.sub(r'http\S+', '', text)
    text = text.strip()
    return text

def chunk_text(text, max_chars=1000, overlap=200):
    """Split long text into overlapping chunks."""
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + max_chars]
        chunks.append(chunk)
        i += (max_chars - overlap)
    return chunks

def process_files():
    all_chunks = []
    for file in tqdm(os.listdir(DATA_DIR)):
        if file.endswith(".txt"):
            with open(os.path.join(DATA_DIR, file), "r", encoding="utf8") as f:
                raw = f.read()
            cleaned = clean_text(raw)
            chunks = chunk_text(cleaned)
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"{file}_{idx}",
                    "source": file,
                    "text": chunk
                })
    with open(OUTPUT_FILE, "w", encoding="utf8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_files()
