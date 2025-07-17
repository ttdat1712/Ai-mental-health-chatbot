import time
from configuration import DEVICE
from model_loader import load_models
from data_processor import initialize_data
from mongo_manager import connect_to_mongodb
from question_suggester import setup_question_suggestion
from answer_generator import generate_answer

def main():
    # Load models
    models = load_models()
    bge_model, bge_tokenizer = models["bge"]
    reranker_model, reranker_tokenizer = models["reranker"]
    qwen_model, qwen_tokenizer = models["qwen"]  # Changed from visstar to qwen

    # Initialize data
    documents_data, bm25, collection_embeddings = initialize_data(bge_model, bge_tokenizer)

    # Setup MongoDB and question suggestion
    connect_to_mongodb()
    setup_question_suggestion()

    # Test queries
    conversation_id = f"test_priority_{int(time.time())}"

    query1 = "Trầm cảm là gì?"
    answer1 = generate_answer(
        conversation_id, query1, qwen_model, qwen_tokenizer,  # Changed from visstar to qwen
        collection_embeddings, bge_model, bge_tokenizer, reranker_model, reranker_tokenizer, bm25,
        alpha=0.6, k_final=3
    )
    print("\n" + "="*50 + "\n")
    time.sleep(1)

    query2 = "Nó có điều trị được không?"
    answer2 = generate_answer(
        conversation_id, query2, qwen_model, qwen_tokenizer,  # Changed from visstar to qwen
        collection_embeddings, bge_model, bge_tokenizer, reranker_model, reranker_tokenizer, bm25,
        alpha=0.5, k_final=3
    )
    print("\n" + "="*50 + "\n")
    time.sleep(1)

if __name__ == "__main__":
    main()s