import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

index = faiss.read_index("evidence.index")

with open("chunks.pkl", "rb") as file:
    documents = pickle.load(file)

def retrieve_evidence(claim, top_k=5):
    query_embedding = model.encode([claim])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):
        similarity_score = 1 / (1 + distance)

        results.append({
            "source": documents[idx]["source"],
            "id": documents[idx]["id"],
            "text": documents[idx]["text"],
            "similarity_score": similarity_score
        })

    return results

def classify_claim(claim):
    evidence = retrieve_evidence(claim)

    top_score = evidence[0]["similarity_score"]

    if top_score >= 0.45:
        classification = "Reliable"
    elif top_score >= 0.30:
        classification = "Questionable"
    else:
        classification = "False or Not Enough Evidence"

    return {
        "claim": claim,
        "classification": classification,
        "top_similarity_score": top_score,
        "evidence": evidence
    }

if __name__ == "__main__":
    claim = input("Enter a medical claim: ")

    result = classify_claim(claim)

    print("\nClaim:")
    print(result["claim"])

    print("\nClassification:")
    print(result["classification"])

    print("\nTop Similarity Score:")
    print(round(result["top_similarity_score"], 3))

    print("\nRetrieved Evidence:")
    for i, item in enumerate(result["evidence"], start=1):
        print(f"\nEvidence {i}")
        print(f"Source: {item['source']}")
        print(f"Abstract ID: {item['id']}")
        print(f"Similarity Score: {round(item['similarity_score'], 3)}")
        print(item["text"][:700] + "...")

