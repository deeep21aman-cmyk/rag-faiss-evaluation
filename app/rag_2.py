import os
from openai import OpenAI

import math

def cosine_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(b*b for b in v2))
    return dot / (mag1 * mag2)


openai_api_key=os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("Set openai api key")
    exit()

client=OpenAI(api_key=openai_api_key)

with open("knowledge.txt",'r') as f:
    text=f.read()

if not text.strip():
    print("File is empty")
    exit()

chunks=[]

for i in range(0,len(text),300):
    chunks.append(text[i:i+300])

chunk_embeddings=[]

for chunk in chunks:
    embedding_response=client.embeddings.create(model= "text-embedding-3-small",input=chunk)
    chunk_embeddings.append((chunk,embedding_response.data[0].embedding))

#print("Chunks:", len(chunks))
#print("Embeddings stored:", len(chunk_embeddings))
#print("Vector length:", len(chunk_embeddings[0][1]))

question=input("How can i help you :\n")

if not question.strip():
    print("question empty")
    exit()

embedded_question=client.embeddings.create(model= "text-embedding-3-small",input=question)
question_vector=embedded_question.data[0].embedding

scores=[]

for chunk,vector in chunk_embeddings:
    score=cosine_similarity(question_vector,vector)
    scores.append((score,chunk))

scores.sort(reverse=True)

top_chunks=[chunk for _,chunk in scores[:2]]

context="\n\n".join(top_chunks)

prompt=f'''
You are answering a question using ONLY the context provided below.

If the answer is not present in the context, reply exactly with:
I don't know.

Return the response in this exact format:

ANSWER:
- one sentence answer grounded in the context

CONTEXT:\n
{context}

QUESTION:\n
{question}
'''

response=client.responses.create(model="gpt-4.1-nano",input=prompt)

print(response.output_text)