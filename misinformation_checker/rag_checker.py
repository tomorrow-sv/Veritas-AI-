import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("evidence.index")

with open("chunks.pkl", "rb") as file:
    documents = pickle.load(file)

def retrieve_evidence(article_text, top_k=3):
    query_embedding = model.encode([article_text])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i in indices[0]:
        results.append(documents[i])

    return results

def classify_article(article_text):
    evidence = retrieve_evidence(article_text)

    evidence_text = " ".join([item["text"] for item in evidence]).lower()
    article_lower = article_text.lower()

    matching_words = 0
    article_words = article_lower.split()

    for word in article_words:
        if word in evidence_text:
            matching_words += 1

    match_ratio = matching_words / max(len(article_words), 1)

    if match_ratio > 0.35:
        classification = "Reliable"
    elif match_ratio > 0.15:
        classification = "Questionable"
    else:
        classification = "False or Not Enough Evidence"

    return classification, evidence