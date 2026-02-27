import numpy as np
import faiss
import math

class VectorStore: #A wrapper around a FAISS index

    def __init__(self):
        self.chunks=[]
        self.index=None

    def load_index(self,file_path):
        self.index=faiss.read_index(file_path)

    def save_index(self,file_path):
        if self.index is None:
            raise Exception ("Index not initialized")
        faiss.write_index(self.index,file_path)
          
    def add(self,new_vectors,new_chunks):
        if not (len(new_chunks)==len(new_vectors)):
            raise ValueError(f"The lengths of vector and chunks is not equal")
        np_vectors=np.array(new_vectors,dtype=np.float32)
        # Compute L2 norm for each row
        norms = np.linalg.norm(np_vectors, axis=1, keepdims=True)
        # Prevent division by zero
        norms[norms == 0] = 1.0
        # Normalize rows
        np_vectors = np_vectors / norms
        if self.index is None:
            dimension=np_vectors.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
           
        self.index.add(np_vectors)
        self.chunks.extend(new_chunks)



    def search(self,query_vector,top_k=3):
        if self.index is None:
            raise Exception ("Indexes not created ")
        np_query=np.array(query_vector,dtype=np.float32)
        np_query = np_query.reshape(1, -1)
        norm=np.linalg.norm(np_query,axis=1,keepdims=True)
        norm[norm==0]=1.0
        np_query=np_query/norm
        
        distances,indices=self.index.search(np_query,top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 - (dist / 2)
            results.append({
                "chunk_id": int(idx),
                "chunk_text": self.chunks[idx]["chunk_text"],
                "similarity": float(similarity)
             })

        return results
