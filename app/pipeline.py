from vector_store import VectorStore
from config import TOP_K,VECTOR_FILE,CHUNKS_FILE,INDEX_FILE_NAME
import loader
import chunker
import storage
import embeddings
import os
import llm
from openai import OpenAI


class RAGPipeline:

    def __init__(self):
        self.store=VectorStore()
        if os.path.exists(INDEX_FILE_NAME):
            text_chunks=storage.load_chunks(CHUNKS_FILE)
            self.store.load_index(INDEX_FILE_NAME)
            self.store.chunks=text_chunks
        else:
            text=loader.load_data()
            text_chunks=chunker.chunk_text(text)
            vector_chunks=embeddings.embed_chunks(text_chunks)
            self.store.add(vector_chunks,text_chunks)
            storage.save_chunks(text_chunks,CHUNKS_FILE)
            self.store.save_index(INDEX_FILE_NAME)

    def ask(self,question: str,use_rerank):
        context = self.search_with_rerank(question)
        return self.generate_answer_from_context(question, context)

        
    def generate_answer_from_context(self,question, context):
        final_context='\n\n'.join(chunk["chunk_text"] for chunk in context)
        return llm.generate_answer(question,final_context)


    def search(self,question: str):
        vector_query=embeddings.embed_query(question)
        context=self.store.search(vector_query,TOP_K)
        return context

    def rerank(self,question,context):
        scores=[]
        for chunk in context:
            score=llm.score_relevance(question,chunk["chunk_text"])
            scores.append((score,chunk))
        scores.sort(key=lambda x: x[0], reverse=True)
        reordered=[chunk for _,chunk in scores]
        return reordered

    def search_with_rerank(self,question):
        context=self.search(question)
        reranked_context=self.rerank(question,context)
        return reranked_context