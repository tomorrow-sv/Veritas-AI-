import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

SOURCE_FOLDER = "trusted_sources"

def load_documents():
    documents = []

    for filename in os.listdir(SOURCE_FOLDER):
        path = os.path.join(SOURCE_FOLDER, filename)

        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        documents.append({
            "source": filename,
            "text": text
        })

    return documents

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = load_documents()

texts = []

for document in documents:
    texts.append(document["text"])

embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "evidence.index")

with open("chunks.pkl", "wb") as file:
    pickle.dump(documents, file)

print("Database created successfully.")