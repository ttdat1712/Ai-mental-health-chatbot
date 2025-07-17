import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def chunk_text_by_hash(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        chunks_raw = re.split(r'\n# ', text)
        chunks = ['# ' + chunk.strip() for chunk in chunks_raw if chunk.strip()]
        print(f"Successfully chunked {len(chunks)} documents from {file_path}")
        return chunks
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error reading or chunking file {file_path}: {e}")
        return []

def min_max_normalize(scores):
    scores_np = np.array(scores).reshape(-1, 1)
    if np.all(scores_np == scores_np[0]):
         return np.full(scores_np.shape, 0.5).flatten()
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(scores_np)
    return normalized_scores.flatten()