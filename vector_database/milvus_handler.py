import configparser
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import json

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['milvus']['uri'], config['milvus']['token']

def connect_milvus():
    uri, token = load_config()
    connections.connect("default", uri=uri, token=token)

def create_collection():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
    ]
    schema = CollectionSchema(fields, "Nvidia CUDA Documentation Embeddings")
    collection = Collection("cuda_docs", schema)
    return collection

def insert_embeddings(collection, embeddings, batch_size=3000):
    ids = [i for i in range(len(embeddings))]
    vectors = [embedding['embedding'] for embedding in embeddings]
    urls = [embedding['url'] for embedding in embeddings]
    contents = [embedding['content'][:3000] for embedding in embeddings]  # Truncate content to 3000 characters

    # Split data into smaller batches
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_vectors = vectors[i:i + batch_size]
        batch_urls = urls[i:i + batch_size]
        batch_contents = contents[i:i + batch_size]
        collection.insert([batch_ids, batch_vectors, batch_urls, batch_contents])

def create_index(collection):
    index_params = {
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
        "metric_type": "L2"
    }
    collection.create_index(field_name="embedding", index_params=index_params)

if __name__ == "__main__":
    connect_milvus()
    collection = create_collection()

    # Insert embeddings only if the collection is new and empty
    if collection.num_entities == 0:
        with open('C:/Users/atulp/Downloads/Steps AI Assignment/cuda-web-crawler-milvus/data_processing/embeddings.json', 'r') as file:
            embeddings = json.load(file)
        insert_embeddings(collection, embeddings)

    # Create index
    create_index(collection)

    # Load collection after creating the index
    collection.load()
