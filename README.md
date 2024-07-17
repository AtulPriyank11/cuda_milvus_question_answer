# Web Crawler and Question Answering System

### Overview

This project consists of a system designed for web crawling, data chunking, vector database creation, retrieval, re-ranking, and question answering. Additionally, it includes a user interface built with Streamlit for user interaction.

### Components

1. Web Crawler: Crawls web pages and extracts content.
2. Data Chunking: Breaks down data into manageable chunks.
3. Vector Database Creation: Creates and stores vector embeddings in a vector database (Milvus).
4. Retrieval and Re-ranking: Retrieves and re-ranks documents based on relevance using hybrid search techniques.
5. Question Answering: Provides answers to questions based on retrieved documents.
6. User Interface: Streamlit-based UI for interacting with the system.

## Setup Instructions
### Prerequisites

1. Python 3.7 or higher
2. pip (Python package installer)
3. Git (for cloning the repository)
4. Milvus instance (For vector database operations) 

### Installation

1. Clone the Repository

```bash
git clone https://github.com/AtulPriyank11/cuda_milvus_question_answer.git
cd cuda_milvus_question_answer
```

2. Create and Activate a Virtual Environment outside of the Repository folder

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install Dependencies which is inside the repo folder

```bash
pip install -r requirements.txt
```


## Zilliz Cloud 

Zilliz Cloud is a fully managed cloud service provided by Zilliz, which offers a scalable and easy-to-use vector database based on Milvus. It simplifies the deployment and management of Milvus databases, providing a user-friendly interface and tools for efficient vector operations, including storage, retrieval, and real-time updates.

## Setting Up Milvus with Zilliz Cloud

> Zilliz Cloud Account: Sign up for an account on Zilliz Cloud if you don’t already have one.
> API Key: Obtain your API key from the Zilliz Cloud dashboard.

### Setup Instructions

1. Create a Milvus Instance on Zilliz Cloud

    Log in to your Zilliz Cloud account.
    Navigate to the Milvus service section.
    Create a new Milvus instance by following the on-screen instructions.
    Configure the instance according to your needs (e.g., select instance type, storage, etc.).

2. Obtain Connection Details

    After creating the Milvus instance, go to the instance details page.
    Note down the URI and API Key provided for your Milvus instance.

3. Configure Your Project

Create or update a config.ini file in the vector_database directory with the following content:

```ini
[milvus]
uri = <YOUR_ZILLIZ_CLOUD_MILVUS_URI>
token = <YOUR_ZILLIZ_CLOUD_API_KEY>
```
4. Connect to Zilliz Cloud Milvus

Ensure you have the pymilvus library installed:

```bash
pip install pymilvus
```
The connection code in your project will use the URI and API key to connect to your Milvus instance:

```python
from pymilvus import connections

def connect_milvus():
    uri, token = load_config()
    connections.connect("default", uri=uri, token=token)
```

### Update Script Parameters

Ensure that paths and parameters in the following scripts are correctly configured:
1. web_crawler.py
2. data_chunking.py
3. vector_database_creation.py
4. retrieval_and_reranking.py
5. question_answering.py
Adjust the configuration to match your environment and Milvus setup.

## Running the System

### Web Crawler

Running the script will crawl the web pages and save the extracted data.

```bash
python web_crawler.py
```

### Data Chunking - Embedding

Running the script will chunk the scraped data and data are converted to embedded vectors.

```bash
python chunking.py
```

```bash
python embeddings.py
```

### Vector Database Creation

Create a vector database and insert embeddings into Milvus.

```bash
python milvus_handler.py
```

Ensure that Milvus is running in Zilliz cloud and properly configured before executing this script.

### Retrieval and Re-Ranking

Perform retrieval and re-ranking of documents based on a query:

```bash
python hybrid_retrieval.py
```

This script will use BM25 and DPR-based methods to retrieve and re-rank documents.

### Question Answering

Generate the answers to questions based on the re-ranked documents.

```bash
python answer_generator.py
```
### User Interface

```bash
streamlit run ui.py
```

## User Interface Usage

### Access the Interface

1. Open your browser and go to http://localhost:8501.
2. You will see the Streamlit application with options to enter queries and view results.

### Interact with the UI
        
1. Enter Query: Type your query in the provided input box.
2. Submit Query: Click the "Submit" button to retrieve and display results.
3. View Results: Results will be displayed on the page, showing relevant documents and answers.

## Notes

1. Ensure you have a running Milvus instance and update the config.ini file with correct URI and token.
2. Modify script parameters as needed to fit your specific use case and environment.
3. Also when running a script, make sure to go to that specific folder and run.
