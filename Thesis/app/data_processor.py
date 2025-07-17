import chromadb
from rank_bm25 import BM25Okapi
from configuration import DATA_FILE_PATH, CHROMA_DB_PATH
from utils import chunk_text_by_hash
from embedding_utils import generate_embeddings

def initialize_data(bge_model, bge_tokenizer):
    # Chunk text
    chunks = chunk_text_by_hash(DATA_FILE_PATH)
    if not chunks:
        print("No data chunks found. Exiting.")
        exit()
    
    # Process documents
    documents_data = []
    corpus_texts = []
    for i, chunk in enumerate(chunks):
        parts = chunk.split("\nNguồn: ")
        text = parts[0].strip()
        if text.startswith("# "):
            text = text[2:]
        source = parts[1].strip() if len(parts) > 1 else "Không có nguồn"
        documents_data.append({"id": str(i), "text": text, "source": source})
        corpus_texts.append(text)
    
    # Initialize BM25
    print("Initializing BM25...")
    try:
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 initialized.")
    except Exception as e:
        print(f"Error initializing BM25: {e}")
        bm25 = None
    
    # Initialize ChromaDB and generate embeddings
    print("Initializing ChromaDB and generating BGE embeddings...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection_name = "mental_health_bge_only_v1"
    collection_embeddings = chroma_client.get_or_create_collection(name=collection_name)
    
    existing_ids = collection_embeddings.get(ids=[d["id"] for d in documents_data])['ids']
    ids_to_add = []
    docs_to_add = []
    embeddings_to_add = []
    metadatas_to_add = []
    
    if len(existing_ids) < len(documents_data):
        print(f"Found {len(existing_ids)} existing BGE embeddings. Generating for missing {len(documents_data) - len(existing_ids)} documents...")
        
        docs_map = {doc["id"]: doc for doc in documents_data}
        texts_to_embed = []
        ids_to_embed = []
        
        for doc_id in [d["id"] for d in documents_data]:
            if doc_id not in existing_ids:
                ids_to_embed.append(doc_id)
                texts_to_embed.append(docs_map[doc_id]["text"])
        
        if texts_to_embed:
            print(f"Generating BGE embeddings for {len(texts_to_embed)} documents...")
            bge_embs = generate_embeddings(texts_to_embed, bge_model, bge_tokenizer)
            final_embeddings = bge_embs.numpy()
            
            for i, doc_id in enumerate(ids_to_embed):
                doc = docs_map[doc_id]
                ids_to_add.append(doc["id"])
                docs_to_add.append(doc["text"])
                embeddings_to_add.append(final_embeddings[i].tolist())
                metadatas_to_add.append({"source": doc["source"]})
            
            if ids_to_add:
                print(f"Adding {len(ids_to_add)} new BGE embeddings to ChromaDB...")
                try:
                    collection_embeddings.add(
                        ids=ids_to_add,
                        embeddings=embeddings_to_add,
                        documents=docs_to_add,
                        metadatas=metadatas_to_add
                    )
                    print("BGE Embeddings added successfully.")
                except Exception as e:
                    print(f"Error adding BGE embeddings to ChromaDB: {e}")
    else:
        print("All BGE embeddings already exist in ChromaDB.")
    
    return documents_data, bm25, collection_embeddings