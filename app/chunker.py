from config import CHUNK_SIZE,CHUNK_OVERLAP

def chunk_text(text):
    text_chunks=[]
    if CHUNK_OVERLAP>CHUNK_SIZE:
        raise ValueError

    chunk_id=0

    for i in range(0,len(text),CHUNK_SIZE-CHUNK_OVERLAP):
        chunk=text[i:i+CHUNK_SIZE]
        chunk_dict={}
        chunk_dict["chunk_id"]=chunk_id
        chunk_dict["chunk_text"]=chunk
        chunk_id=chunk_id+1
        text_chunks.append(chunk_dict)
    return text_chunks