import faiss
import numpy as np

def build_faiss_index(embeddings, embedding_size):
    nlist = 50
    m = 32
    quantizer = faiss.IndexFlatL2(embedding_size)
    index = faiss.IndexIVFPQ(quantizer, embedding_size, nlist, m, 8)
    index.train(embeddings)
    index.add(embeddings)
    return index

def search_faiss_index(index, query_embeddings, k):
    index.nprobe = 10
    distances, indices = index.search(query_embeddings.astype(np.float32), k)
    return distances, indices