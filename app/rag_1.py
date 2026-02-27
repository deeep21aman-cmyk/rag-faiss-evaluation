import os
from openai import OpenAI

openai_api_key=os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Set API key")
    exit()

client=OpenAI(api_key=openai_api_key)

with open("knowledge.txt",'r') as f:
    document_text=f.read()

chunks_list=[]

for i in range(0,len(document_text),300):
    chunks_list.append(document_text[i:i+300])

question=input("Ask a question : ")

words_in_question=question.split(" ")

relevant_chunks=[]

for chunk in chunks_list:
    for ques in words_in_question:
        if ques.lower() in chunk.lower():
            if chunk not in relevant_chunks:
                relevant_chunks.append(chunk)   

if len(relevant_chunks)==0:
    print("No relevenat info found")
    exit()
else:
    context=' '.join(relevant_chunks)

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