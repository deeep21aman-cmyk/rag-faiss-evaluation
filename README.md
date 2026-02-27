⚠️ Requires OpenAI API key set as environment variable:
export OPENAI_API_KEY=your_key_here
# Retrieval-Augmented Generation System (FAISS + Evaluation)
Overview

This project implements a Retrieval-Augmented Generation (RAG) system built from scratch using:
	•	OpenAI embeddings
	•	FAISS for vector similarity search
	•	Cosine similarity via L2 normalization
	•	LLM-based reranking
	•	Full retrieval and generation evaluation metrics
	•	Index persistence for production-like behavior

The goal of this project was to understand and implement the full RAG lifecycle — from embedding generation and vector indexing to evaluation and performance tradeoff analysis.

⸻

System Architecture

The system is divided into clean, modular layers:

1️⃣ VectorStore
	•	Uses FAISS IndexFlatL2
	•	Stores normalized embeddings
	•	Persists index to disk
	•	Performs top-K vector retrieval

2️⃣ Retrieval Layer
	•	Embeds incoming queries
	•	Searches FAISS index
	•	Returns top-K chunk matches

3️⃣ Reranking Layer (Optional)
	•	Uses LLM scoring to reorder retrieved chunks
	•	Improves ranking precision (Top-1, MRR)
	•	Adds measurable latency cost

4️⃣ Generation Layer
	•	Constructs context from retrieved chunks
	•	Sends grounded prompt to LLM
	•	Produces final answer

5️⃣ Evaluation Framework
	•	Measures:
	•	Top-1 Accuracy
	•	Top-K Accuracy
	•	Mean Reciprocal Rank (MRR)
	•	Answer Accuracy
	•	Retrieval Latency
	•	Generation Latency
	•	Total Latency

The evaluator explicitly controls retrieval mode (with or without rerank) to ensure clean and reproducible experiments.

⸻

Indexing Strategy

Why Normalize Embeddings?

Embeddings are L2-normalized before indexing.
This ensures:
	•	All vectors lie on the unit sphere.
	•	L2 distance becomes equivalent to cosine similarity.
	•	Consistent similarity comparisons.

Mathematically:

If vectors are normalized:
||u - v||² = 2 - 2 * cos(u, v)

Thus minimizing L2 distance ≈ maximizing cosine similarity.

Why FAISS IndexFlatL2?
	•	Exact nearest neighbor search.
	•	Simple, transparent behavior.
	•	Appropriate for small datasets.
	•	Good baseline for experimentation before moving to approximate or managed vector databases.

Index Persistence

The FAISS index is saved to disk after creation and loaded on startup if available.
This avoids re-embedding and rebuilding the index on every run.

⸻

Experimental Results

Evaluation was performed on predefined test cases.

Without Rerank
	•	Top-1 Accuracy: 0.4
	•	Top-K Accuracy: 1.0
	•	MRR: 0.67
	•	Answer Accuracy: 0.8
	•	Avg Total Latency: ~1.24s

With Rerank
	•	Top-1 Accuracy: 0.6
	•	MRR: 0.8
	•	Answer Accuracy: 0.8
	•	Avg Total Latency: ~3.37s

⸻

Key Observations
	•	Reranking improved ranking precision (Top-1 and MRR).
	•	Answer accuracy did not improve because the correct chunk was already present in Top-K retrieval.
	•	Reranking significantly increased latency (~3x total time).
	•	This demonstrates a real-world tradeoff between precision and performance.

⸻

Engineering Lessons Learned
	1.	Evaluation must explicitly control retrieval mode — global flags introduce hidden coupling.
	2.	Reranking improves ranking quality but may not improve final answer quality.
	3.	Strict grounding prompts reduce hallucination but can cause false negatives.
	4.	Measuring latency is critical when adding additional LLM calls.
	5.	Clean separation of retrieval, reranking, and generation enables reproducible experiments.

⸻

Future Improvements
	•	Hybrid retrieval (keyword + vector)
	•	Approximate nearest neighbor indexing
	•	Managed vector database integration (Pinecone / Weaviate / Qdrant)
	•	Larger-scale benchmarking
	•	API deployment layer

⸻

How to Run
	1.	Set OpenAI API key.
	2.	Run indexing (first run builds FAISS index).
	3.	Run evaluation or interactive mode.

⸻

Why This Project Matters

This project demonstrates:
	•	Understanding of vector similarity search internals
	•	Knowledge of cosine vs L2 behavior
	•	Retrieval quality evaluation using IR metrics
	•	Reranking tradeoff analysis
	•	Clean system design principles
	•	Production-like index persistence

It is intentionally implemented without high-level RAG frameworks to ensure full control and understanding of each layer.
