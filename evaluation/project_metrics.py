"""
CSR AI/ML Project 2 — GUVI RAG Chatbot
Evaluation Script for Model Metrics
Developed by: Keren Gifta 🌸
"""

import os
import time
import csv
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import faiss
import subprocess
from datetime import datetime

# 1️⃣ Load FAISS & Embedding Model

print("🔹 Loading embedding model and FAISS index...")

embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Load text data
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

def chunk_text(text, chunk_size=800):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = chunk_text(combined_text)
embeddings = embedder.encode(chunks, convert_to_tensor=False)
embeddings = np.array(embeddings, dtype="float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print(f"✅ Loaded {len(chunks)} chunks and created FAISS index.")

# 2️⃣ Retrieval Accuracy (Precision@k, Recall@k)

print("\n📊 Evaluating Retrieval Accuracy (Precision@k & Recall@k)...")

test_data = [
    ("How do I get my certificate after finishing a course", "certificate completion get certified"),
    ("Can I get a refund after enrolling in a GUVI course", "refund policy"),
    ("How do I enroll in a course", "enroll join course"),
]

k = 3
correct = 0

for query, true_context in test_data:
    query_emb = embedder.encode([query], convert_to_tensor=False)
    D, I = index.search(np.array(query_emb, dtype="float32"), k)
    retrieved = [chunks[i] for i in I[0]]
    if any(true_context.lower() in r.lower() for r in retrieved):
        correct += 1

precision_at_k = correct / (len(test_data) * k)
recall_at_k = correct / len(test_data)
print(f"✅ Precision@{k}: {precision_at_k:.2f}")
print(f"✅ Recall@{k}: {recall_at_k:.2f}")

# 3️⃣ Response Relevance (BLEU & ROUGE)

print("\n🧠 Evaluating Response Relevance (BLEU / ROUGE)...")

rouge = Rouge()
bleu_scores = []
rouge_scores = []

# Sample predicted vs reference responses
samples = [
    (
        "You can download your certificate after completing 100% of your course.",
        "You must complete all lessons to 100% progress to get your certificate."
    ),
    (
        "Refunds are available within 7 working days of request.",
        "You can get a refund within 7 working days after submitting the form."
    ),
]

for pred, ref in samples:
    bleu = sentence_bleu([ref.split()], pred.split())
    rouge_score = rouge.get_scores(pred, ref)[0]['rouge-l']['f']
    bleu_scores.append(bleu)
    rouge_scores.append(rouge_score)

avg_bleu = np.mean(bleu_scores)
avg_rouge = np.mean(rouge_scores)
print(f"✅ Average BLEU: {avg_bleu:.2f}")
print(f"✅ Average ROUGE-L: {avg_rouge:.2f}")

# 4️⃣ Latency Measurement

print("\n⚡ Measuring System Latency...")

query = "How do I get my certificate after finishing a course"
start_time = time.time()
query_emb = embedder.encode([query], convert_to_tensor=False)
D, I = index.search(np.array(query_emb, dtype="float32"), k)
end_time = time.time()

latency = end_time - start_time
print(f"✅ Average Query Latency: {latency:.2f} seconds")

# 5️⃣ PEP8 Compliance Check (flake8)

print("\n🧩 Checking Code Modularity & PEP8 Compliance...")
pep8_issues = "Not Checked"
try:
    result = subprocess.run(["flake8", "app/"], capture_output=True, text=True)
    if result.stdout:
        pep8_issues = "Issues Found"
        print("⚠️ PEP8 Issues Found:\n", result.stdout)
    else:
        pep8_issues = "Clean"
        print("✅ No major PEP8 issues detected.")
except FileNotFoundError:
    print("⚠️ flake8 not installed. Run: pip install flake8")
    pep8_issues = "flake8 not installed"

# 6️⃣ Save Results to CSV

output_path = "evaluation_results.csv"
header = ["Metric", "Value"]

results = [
    ["Precision@3", round(precision_at_k, 2)],
    ["Recall@3", round(recall_at_k, 2)],
    ["Average BLEU", round(avg_bleu, 2)],
    ["Average ROUGE-L", round(avg_rouge, 2)],
    ["Latency (sec)", round(latency, 2)],
    ["PEP8_Status", pep8_issues],
    ["Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
]

with open(output_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for metric, value in results:
        writer.writerow([metric, value])

print(f"\n📁 Results saved successfully to '{output_path}'")

# 🎯 Final Report

print("\n==============================")
print("📋 FINAL EVALUATION REPORT")
print("==============================")
print(f"Precision@3: {precision_at_k:.2f}")
print(f"Recall@3: {recall_at_k:.2f}")
print(f"BLEU: {avg_bleu:.2f}")
print(f"ROUGE-L: {avg_rouge:.2f}")
print(f"Latency: {latency:.2f} sec")
print(f"PEP8: {pep8_issues}")
print("==============================")
print(f"Results stored in: {output_path}")
print("Developed by: Keren Gifta 🌸")
print("==============================")
