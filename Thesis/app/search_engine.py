import time
import torch
import numpy as np
from utils import min_max_normalize
from embedding_utils import generate_embeddings

def hybrid_search_and_rerank(query, collection_embeddings, bge_model, bge_tokenizer, reranker_model, reranker_tokenizer, bm25=None, alpha=0.5, k_embed_retrieval=50, top_k_initial=5, top_k_final=3):
    start_time = time.time()
    results_data = {}
    
    # 1. BGE Embedding Search
    try:
        print(f"Step 1: Performing BGE Embedding Search (k_embed_retrieval={k_embed_retrieval})...")
        query_embedding_bge = generate_embeddings([query], bge_model, bge_tokenizer)[0]
        query_embedding_final = query_embedding_bge.tolist()
        
        embed_results = collection_embeddings.query(
            query_embeddings=[query_embedding_final],
            n_results=k_embed_retrieval,
            include=['metadatas', 'documents', 'distances']
        )
        
        if not embed_results or not embed_results.get("ids") or not embed_results["ids"][0]:
            print("Warning: BGE Embedding search returned no results.")
            return []
        
        for i, doc_id in enumerate(embed_results["ids"][0]):
            distance = embed_results["distances"][0][i]
            similarity = 1.0 - distance
            results_data[doc_id] = {
                "id": doc_id,
                "text": embed_results["documents"][0][i],
                "source": embed_results["metadatas"][0][i].get("source", "Không có nguồn"),
                "embedding_score": similarity,
                "bm25_score": 0.0,
                "combined_score": 0.0,
                "rerank_score": -float('inf')
            }
        print(f"Found {len(results_data)} candidates from BGE embedding search.")
    
    except Exception as e:
        print(f"Error during BGE embedding search: {e}")
        return []
    
    # 2. BM25 Search
    if bm25 and results_data:
        try:
            print("Step 2: Calculating BM25 scores for embedding candidates...")
            query_tokens = query.lower().split()
            candidate_indices = [int(doc_id) for doc_id in results_data.keys()]
            candidate_bm25_scores = bm25.get_scores(query_tokens)
            
            max_bm25_score = 0
            for doc_id in results_data.keys():
                doc_index = int(doc_id)
                if 0 <= doc_index < len(candidate_bm25_scores):
                    score = candidate_bm25_scores[doc_index]
                    results_data[doc_id]["bm25_score"] = score
                    if score > max_bm25_score: max_bm25_score = score
                else:
                    print(f"Warning: Document index {doc_index} out of bounds for BM25 scores.")
                    results_data[doc_id]["bm25_score"] = 0.0
        except Exception as e:
            print(f"Error during BM25 calculation: {e}")
    
    # 3. Combine Scores and Initial Ranking
    print(f"Step 3: Combining scores (alpha={alpha}) and selecting top {top_k_initial}...")
    candidates = list(results_data.values())
    if not candidates:
        print("No candidates to combine or rank.")
        return []
    
    embedding_scores = [c["embedding_score"] for c in candidates]
    bm25_scores = [c["bm25_score"] for c in candidates]
    
    normalized_embedding = min_max_normalize(embedding_scores)
    normalized_bm25 = min_max_normalize(bm25_scores) if bm25 else np.zeros(len(candidates))
    
    for i, candidate in enumerate(candidates):
        candidate["combined_score"] = (alpha * normalized_embedding[i]) + ((1 - alpha) * normalized_bm25[i])
    
    candidates.sort(key=lambda x: x["combined_score"], reverse=True)
    initial_top_k = candidates[:top_k_initial]
    
    if not initial_top_k:
        print("No candidates after initial ranking.")
        return []
    print(f"Selected {len(initial_top_k)} candidates after initial ranking.")
    
    # 4. Reranking
    print(f"Step 4: Reranking top {len(initial_top_k)} candidates...")
    rerank_pairs = [[query, candidate['text']] for candidate in initial_top_k]
    if rerank_pairs:
        try:
            with torch.no_grad():
                inputs = reranker_tokenizer(rerank_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(bge_model.device)  # Dùng device từ bge_model
                scores = reranker_model(**inputs, return_dict=True).logits.view(-1).float()
                rerank_scores = torch.sigmoid(scores).cpu().numpy()
            
            for i, candidate in enumerate(initial_top_k):
                candidate["rerank_score"] = rerank_scores[i]
        except Exception as e:
            print(f"Error during reranking: {e}")
    
    initial_top_k.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    # 5. Final Selection
    final_results = initial_top_k[:top_k_final]
    end_time = time.time()
    print(f"Step 5: Selected final {len(final_results)} results. Total time: {end_time - start_time:.2f}s")
    
    for res in final_results:
        res["embedding_score"] = float(res["embedding_score"])
        res["bm25_score"] = float(res["bm25_score"])
        res["combined_score"] = float(res["combined_score"])
        res["rerank_score"] = float(res["rerank_score"])
    
    return final_results