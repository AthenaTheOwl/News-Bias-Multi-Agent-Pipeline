# vectorstore/faiss_store.py

import faiss
import os
from sentence_transformers import SentenceTransformer
import numpy as np

INDEX_PATH = "cache/faiss_index/articles.index"
METADATA_PATH = "cache/faiss_index/articles_meta.npy"

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_faiss_index_and_filter(articles):
    """
    Takes a list of {'title', 'link'} dicts.
    Embeds titles and filters out semantically duplicate ones using FAISS.
    Returns a deduplicated list.
    """
    if not articles:
        return []

    # Embed article titles
    titles = [a["title"] for a in articles]
    embeddings = model.encode(titles)

    # Load or create FAISS index
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        metadata = np.load(METADATA_PATH, allow_pickle=True).tolist()
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        metadata = []

    filtered_articles = []
    for i, embed in enumerate(embeddings):
        D, I = index.search(np.array([embed]), k=1)
        if I[0][0] == -1 or D[0][0] > 0.6:
            # Not too close — accept it
            filtered_articles.append(articles[i])
            index.add(np.array([embed]))
            metadata.append(articles[i])
        else:
            print(f"🛑 Skipping near-duplicate: {articles[i]['title']}")

    # Save updated index
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(METADATA_PATH, metadata)

    return filtered_articles
