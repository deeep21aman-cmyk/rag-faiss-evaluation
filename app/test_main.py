from loader import load_data   # your raw text loader
import chunker    # your chunking function
from embeddings import embed_chunks,embed_query  # your embedding generator
from vector_store import VectorStore


def main():
    # 1️⃣ Load raw knowledge text
    text = load_data()

    # 2️⃣ Chunk the text
    chunks = chunker.chunk_text(text)

    print(f"Total chunks created: {len(chunks)}")

    # 3️⃣ Generate embeddings for chunks
    vectors = embed_chunks(chunks)

    print("Embeddings generated.")

    # 4️⃣ Initialize VectorStore and add data
    vs = VectorStore()
    vs.add(vectors, chunks)

    print("FAISS index built.\n")

    # 5️⃣ Ask a real query
    query = "Who invented the telephone?"
    query_vector = embed_query(query)

    # 6️⃣ Search
    results = vs.search(query_vector, top_k=3)

    # 7️⃣ Print results
    print("Top Results:\n")
    for r in results:
        print(f"Chunk ID: {r['chunk_id']}")
        print(f"Similarity: {r['similarity']:.4f}")
        print(f"Text Preview: {r['chunk_text'][:200]}")
        print("-" * 60)


if __name__ == "__main__":
    main()