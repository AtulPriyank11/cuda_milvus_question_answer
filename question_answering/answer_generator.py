import json
import sys
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline

# Add the path to the retrieval folder
sys.path.insert(0, str(Path(__file__).resolve().parent / '../retrieval'))

from hybrid_retrieval import hybrid_search, connect_milvus

# Load the DistilBERT model and tokenizer fine-tuned for QA
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

# Initialize the question-answering pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

if __name__ == "__main__":
    connect_milvus()

    with open('../retrieval/retrieved_results.json', 'r') as file:
        documents = json.load(file)

    question = "What is CUDA?"
    search_results = hybrid_search(question, documents)

    if not search_results:
        print("No search results found.")
        context = ""
    else:
        context_parts = []
        for result in search_results:
            if isinstance(result, dict):
                content = result.get('content', '')
                if content:
                    context_parts.append(content)
                else:
                    print(f"Skipping document with ID {result.get('id', 'unknown')} due to empty content.")
            else:
                print(f"Unexpected format for result: {result}")
        
        context = " ".join(context_parts)
        # Truncate context to fit the maximum input length for DistilBERT
        context = context[:1000]  # DistilBERT can handle inputs up to 512 tokens, roughly 1000 characters

    print("Context:", context)
    
    if context:
        answer = answer_question(question, context)
        print("Answer:", answer)
    else:
        print("Cannot generate an answer: context is empty.")
