import json
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load scraped data
with open('C:/Users/atulp/Downloads/Steps AI Assignment/cuda-web-crawler-milvus/web_crawler/nvidia_docs.json', 'r') as f:
    data = json.load(f)

# Preprocess text
def preprocess(text):
    return [token for token in simple_preprocess(text)]

# Tokenize content and build dictionary
documents = [preprocess(item['content']) for item in data]
dictionary = Dictionary(documents)
corpus = [dictionary.doc2bow(doc) for doc in documents]

# Train LDA model for topic modeling
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, random_state=42)

# Get topic distributions for documents
topics = [lda_model.get_document_topics(bow, minimum_probability=0) for bow in corpus]

# Convert topic distributions to vectors
def topic_to_vector(topic_dist, num_topics):
    vector = np.zeros(num_topics)
    for topic, prob in topic_dist:
        vector[topic] = prob
    return vector

topic_vectors = [topic_to_vector(topic, lda_model.num_topics) for topic in topics]

# Create chunks based on topic similarity
def create_chunks_by_topics(documents, topic_vectors, threshold=0.5):
    chunks = []
    current_chunk = [documents[0]]
    current_vector = topic_vectors[0]
    
    for doc, vector in zip(documents[1:], topic_vectors[1:]):
        similarity = cosine_similarity([current_vector], [vector])[0][0]
        if similarity >= threshold:
            current_chunk.append(doc)
        else:
            chunks.append(current_chunk)
            current_chunk = [doc]
            current_vector = vector
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

chunks = create_chunks_by_topics(data, topic_vectors)

chunked_data = [
    {
        'url': chunk[0]['url'],
        'chunks': ' '.join([doc['content'] for doc in chunk])
    } for chunk in chunks
]

with open('nvidia_docs_chunks.json', 'w') as f:
    json.dump(chunked_data, f, indent=4)
