from openai import OpenAI
from config import OPEN_AI_MODEL
import os

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("set API KEY")


client=OpenAI(api_key=OPENAI_API_KEY)

def generate_answer(question:str,context_chunks:list[str])->str:
    prompt=f'''
You are answering a question using ONLY the context provided below.

Base your answer strictly on the context. Extract relevant facts directly from it. Do not add outside knowledge.

Return the response in this exact format:

ANSWER:
- one sentence answer grounded in the context

CONTEXT:
{"\n\n".join(context_chunks)}

QUESTION:\n
{question}
'''
    response=client.responses.create(model=OPEN_AI_MODEL,input=prompt)
    
    return response.output_text
    

def score_relevance(question,chunk_text):
    prompt=f"Given this question and this chunk, rate relevance from 0 to 10.\nOnly return a number.\nQuestion:\n{question}\n\nChunk:\n{chunk_text}"
    response=client.responses.create(model=OPEN_AI_MODEL,input=prompt)
    text=response.output_text.strip()
    try:
        for token in text.split():
            if token.isdigit():
                score= int(token)
                return max(0,min(score,10))
        return 0
    except Exception:
        return 0

