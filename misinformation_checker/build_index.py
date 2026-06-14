import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

SOURCE_FOLDER = "trusted_sources"

def load_documents():
    documents = []

    for filename in os.listdir(SOURCE_FOLDER):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(SOURCE_FOLDER, filename)

        with open(path, "r", encoding="utf-8") as file:
            content = file.read()

        abstracts = content.split("###")

        for abstract in abstracts:
            abstract = abstract.strip()

            if not abstract:
                continue

            lines = abstract.splitlines()

            abstract_id = lines[0]

            text = []

            for line in lines[1:]:

                if "\t" in line:
                    section, sentence = line.split("\t", 1)
                    text.append(sentence)

            documents.append({
                "source": filename,
                "id": abstract_id,
                "text": " ".join(text)
            })

    return documents

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = load_documents()

texts = []

for document in documents:
    texts.append(document["text"])

print(f"Loaded {len(documents)} medical abstracts.")

embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index contains {index.ntotal} documents.")
print(f"Embedding shape: {embeddings.shape}")

faiss.write_index(index, "evidence.index")

with open("chunks.pkl", "wb") as file:
    pickle.dump(documents, file)

print("Database created successfully.")