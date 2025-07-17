import chromadb
import re
from chromadb.utils import embedding_functions
from configuration import CHROMA_QUESTION_DB_PATH, BGE_MODEL_NAME, QUESTION_FILE_PATH

def setup_question_suggestion():
    print("Setting up question suggestion...")
    try:
        chroma_client_q = chromadb.PersistentClient(path=CHROMA_QUESTION_DB_PATH)
        q_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=BGE_MODEL_NAME)
        collection_question = chroma_client_q.get_or_create_collection(
            name="suggested_questions_bge_v1",
            embedding_function=q_embedding_function
        )
        with open(QUESTION_FILE_PATH, "r", encoding="utf-8") as f:
            question_list = [line.strip() for line in f if line.strip()]
        existing_q_ids = collection_question.get(ids=[str(i) for i in range(len(question_list))])['ids']
        questions_to_add = []
        q_ids_to_add = []
        if len(existing_q_ids) < len(question_list):
            print(f"Adding {len(question_list) - len(existing_q_ids)} missing questions for suggestion...")
            for i, q in enumerate(question_list):
                q_id = str(i)
                if q_id not in existing_q_ids:
                    q_ids_to_add.append(q_id)
                    questions_to_add.append(q)
            if q_ids_to_add:
                try:
                    collection_question.add(ids=q_ids_to_add, documents=questions_to_add)
                    print(f"Added {len(q_ids_to_add)} questions.")
                except Exception as e:
                    print(f"Error adding questions: {e}")
        else:
            print("All suggested questions already exist.")
        return collection_question
    except FileNotFoundError:
        print(f"Error: Question file not found: {QUESTION_FILE_PATH}")
        return None
    except Exception as e:
        print(f"Error setting up question suggestion: {e}")
        return None

collection_question = setup_question_suggestion()

def clean_question(question):
    return re.sub(r"^Câu \d+: ", "", question).strip()

def suggest_questions(query_text, history, top_k=3):
    if not collection_question:
        return []
    try:
        results = collection_question.query(query_texts=[query_text], n_results=top_k + 5)
        suggested_questions_raw = results["documents"][0] if results["documents"] else []
        asked_questions_texts = {msg["text"].strip().lower() for msg in history if msg["role"] == "user"}
        asked_questions_texts.add(query_text.strip().lower())
        filtered_questions = []
        for q_raw in suggested_questions_raw:
            q_clean = clean_question(q_raw)
            if q_clean.strip().lower() not in asked_questions_texts:
                filtered_questions.append(q_clean)
                if len(filtered_questions) == top_k:
                    break
        return filtered_questions
    except Exception as e:
        print(f"Error suggesting questions: {e}")
        return []