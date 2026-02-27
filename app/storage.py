import numpy as np
import json

def save_vectors(vectors,file_path):
    array=np.array(vectors)
    np.save(file_path,array)

def load_vectors(file_path):
    array=np.load(file_path)
    return array.tolist()

def save_chunks(chunks,file_path):
    with open(file_path,'w') as f:
        json.dump(chunks,f)

def load_chunks(file_path):
    with open(file_path,'r') as f:
        return json.load(f)