import chromadb
from chromadb.utils import embedding_functions
import re
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from datetime import datetime
import time

HUGGINGFACE_TOKEN = "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"

# Original model paths
BGE_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Updated fine-tuned model paths
QWEN_MODEL_NAME = "/content/drive/MyDrive/Thesis/finetune_qwen/qwen_finetune"
LLAMA_MODEL_NAME = "/content/drive/MyDrive/Thesis/finetune_llama/llama_finetune"
GEMMA_MODEL_NAME = "/content/drive/MyDrive/Thesis/finetune_gemma/finetune_gemma"

AVAILABLE_MODELS = {
    "Qwen 2.5B (Fine-tuned)": QWEN_MODEL_NAME,
    "Llama 3.2B (Fine-tuned)": LLAMA_MODEL_NAME,
    "Gemma 2B (Fine-tuned)": GEMMA_MODEL_NAME
}

DATA_FILE_PATH = "/content/drive/MyDrive/Thesis/data/data.txt"
QUESTION_FILE_PATH = "/content/drive/MyDrive/Thesis/data/question.txt"

# ChromaDB paths
CHROMA_DB_PATH = "./chroma_db_bge"
CHROMA_QUESTION_DB_PATH = "./chroma_question_db"

# MongoDB configurations
MONGO_URI = "mongodb+srv://ttdat11a13:1234@cluster0.qvw7p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_DB_NAME = "chatbot"
MONGO_COLLECTION_NAME = "chat_bot_history_bge"

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")