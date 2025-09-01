import faiss
import os
import json
from sentence_transformers import SentenceTransformer

class FAISSSearch:
    def __init__(self, index_dir="cache/faiss_index"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.meta_path = os.path.join(index_dir, "meta.json")
        self.index = faiss.IndexFlatL2(384)  # 384 dim for MiniLM
        self.metadata = []

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def add_document(self, text: str, meta: dict):
        vector = self.model.encode([text])
        self.index.add(vector)
        self.metadata.append(meta)
        self._save()

    def search(self, query: str, k=5):
        vector = self.model.encode([query])
        scores, indices = self.index.search(vector, k)
        return "\n\n".join([f"Result {i+1}: {self.metadata[i]['headline']}\nURL: {self.metadata[i]['url']}\nBias: {self.metadata[i]['bias']}\n" for i in indices[0] if i < len(self.metadata)])

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
