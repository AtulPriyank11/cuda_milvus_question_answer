import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
import configparser
import numpy as np

def load_config():
    config = configparser.ConfigParser()
    config.read('../vector_database/config.ini')
    return config['milvus']['uri'], config['milvus']['token']

def connect_milvus():
    uri, token = load_config()
    connections.connect("default", uri=uri, token=token)

def fetch_documents_from_milvus(collection_name, limit=1000):
    collection = Collection(collection_name)
    expr = ""  # Empty string retrieves all documents
    all_docs = collection.query(expr=expr, output_fields=["id", "url", "content", "embedding"], limit=limit)
    return all_docs

def query_expansion(query: str) -> str:
    # Improved query expansion with domain-specific terms
    expanded_query = query + " CUDA programming GPU optimization parallel computing NVIDIA"
    return expanded_query

def search_bm25(query: str, documents: list, top_k=10):
    bm25 = BM25Okapi([doc['content'].split() for doc in documents])
    scores = bm25.get_scores(query.split())
    sorted_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_k]

def search_dpr(query: str, top_k=10):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)
    search_params = {"metric_type": "L2", "params": {"ef": 128}}
    collection = Collection("cuda_docs")
    results = collection.search([query_embedding.tolist()], "embedding", search_params, limit=top_k, output_fields=["id", "url", "content", "embedding"])
    return results[0]

def hybrid_search(query: str, documents: list, top_k=10):
    expanded_query = query_expansion(query)
    
    # Perform BM25 search
    bm25_results = search_bm25(expanded_query, documents, top_k=top_k)
    
    # Perform DPR search
    dpr_results = search_dpr(expanded_query, top_k=top_k)
    
    # Prepare results for merging
    combined_results = []
    for doc, bm25_score in bm25_results:
        combined_results.append({
            "id": doc['id'],
            "url": doc['url'],
            "content": doc['content'],
            "embedding": doc['embedding'],
            "score": bm25_score,
            "source": "BM25"
        })
    
    for result in dpr_results:
        combined_results.append({
            "id": result.id,
            "url": result.entity.get("url"),
            "content": result.entity.get("content"),
            "embedding": result.entity.get("embedding"),
            "score": 1.0 / (result.distance + 1e-5),  # Inverse distance as score
            "source": "DPR"
        })
    
    # Re-rank results based on combined scores
    combined_results.sort(key=lambda x: x['score'], reverse=True)
    
    return combined_results[:top_k]

if __name__ == "__main__":
    connect_milvus()

    # Load the document embeddings from Milvus collection
    documents = fetch_documents_from_milvus("cuda_docs")

    # Define a query
    query = "What is CUDA?"
    
    # Perform a hybrid search
    results = hybrid_search(query, documents)

    # Save the search results to a JSON file
    with open('retrieved_results.json', 'w') as file:
        json.dump([{
            "id": result["id"],
            "url": result["url"],
            "content": result["content"],
            "embedding": result["embedding"],
            "score": result["score"],
            "source": result["source"]
        } for result in results], file, indent=4, default=str)  # Use default=str to handle non-serializable types

    # Print the search results
    for result in results:
        print(f"ID: {result['id']}, URL: {result['url']}, Content: {result['content']}, Embedding: {result['embedding']}, Score: {result['score']}, Source: {result['source']}")
