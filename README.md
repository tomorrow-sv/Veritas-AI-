# VeritasAI

## Overview

VeritasAI is a prototype medical misinformation detection system that uses Retrieval-Augmented Generation (RAG) concepts, semantic search, and evidence retrieval to evaluate medical claims. Instead of relying on keyword matching, the system searches for semantically similar medical research and returns supporting evidence along with a credibility classification.

---

## System Purpose

The goal of VeritasAI is to help users evaluate medical claims by retrieving relevant scientific evidence from published medical abstracts.

The system:
- Accepts a medical claim as input.
- Uses semantic search to find similar medical research.
- Retrieves the most relevant abstracts.
- Classifies the claim as **Reliable**, **Questionable**, or **False / Not Enough Evidence** based on evidence strength.
- Displays the retrieved evidence to support the classification.

This project is intended as a **decision-support tool** and does **not** replace professional medical advice.

---

## Data Source

VeritasAI uses the **PubMed 200K RCT Dataset**, which contains approximately 200,000 randomized controlled trial abstracts from biomedical literature.

Each abstract includes:
- PubMed Abstract ID
- Abstract text
- Sentence labels (Background, Objective, Method, Result, Conclusion)

The system converts each abstract into sentence embeddings and stores them in a FAISS vector database for fast semantic retrieval.

---

## Technologies Used

- Python
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS
- NumPy
- Pickle

---

## Limitations

- The system retrieves semantically similar evidence but does not perform full fact verification.
- Classifications are based on semantic similarity thresholds rather than expert medical judgment.
- Paper titles and author information are not included in the current dataset.
- Retrieved evidence may be related to a claim without fully supporting or contradicting it.
- Results should not be used as a substitute for professional medical advice.

---

## Future Improvements

- Add Natural Language Inference (NLI) for stronger fact verification.
- Display paper titles and authors.
- Improve confidence scoring.
- Develop a web-based user interface.
- Support additional trusted medical data sources.