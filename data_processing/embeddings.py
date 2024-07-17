import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    return model.encode(text).tolist()

if __name__ == "__main__":
    with open('nvidia_docs_chunks.json', 'r') as file:
        chunks = json.load(file)

    embeddings = []
    for chunk in chunks:
        embeddings.append({
            'id': chunks.index(chunk),  # Adding id for Milvus
            'url': chunk['url'],
            'content': chunk['chunks'],
            'embedding': embed_text(chunk['chunks'])
        })

    with open('embeddings.json', 'w') as file:
        json.dump(embeddings, file, indent=4)
