import math
class VectorStore:

    def __init__(self):
        self.vectors=[]
        self.chunks=[]

    @staticmethod
    def cosine_similarity(vector_query,vector_chunk):
        if len(vector_query)!=len(vector_chunk):
            raise ValueError("Embedding length should be same")
        dot_product=0
        magnitude_query=0
        magnitude_chunk=0
        for i in range(0,len(vector_query)):
            dot_product=dot_product+(vector_query[i]*vector_chunk[i])
            magnitude_query=magnitude_query+(vector_query[i]*vector_query[i])
            magnitude_chunk=magnitude_chunk+(vector_chunk[i]*vector_chunk[i])
        magnitude_query=math.sqrt(magnitude_query)
        magnitude_chunk=math.sqrt(magnitude_chunk)
        if magnitude_query==0 or magnitude_chunk==0:
            return 0
        
        return dot_product/(magnitude_chunk*magnitude_query)
            
    def add(self,new_vectors,new_chunks):
        if not (len(new_chunks)==len(new_vectors)):
            raise ValueError(f"The lengths of vector and chunks is not equal")

        self.chunks.extend(new_chunks)
        self.vectors.extend(new_vectors)

    def search(self,query_vector,top_k=3):
        if not self.vectors:
            return []
        if len(query_vector)!=len(self.vectors[0]):
            raise ValueError("Embedding for query and stored vector should be same")
        
        scores=[]
        return_data=[]

        for index,knowledge_vector in enumerate(self.vectors):
            score=self.cosine_similarity(query_vector,knowledge_vector)
            scores.append((score,index))
            
        scores.sort(reverse=True)
        top_scores=scores[:top_k]
        for score,index in top_scores:
            return_dict={}
            return_dict["chunk_id"]=self.chunks[index]["chunk_id"]
            return_dict["chunk_text"]=self.chunks[index]["chunk_text"]
            return_dict["similarity_score"]=score
            return_data.append(return_dict)

        return return_data