import os
import re
import pickle

class InvertedIndex:
    def __init__(self):
        # token -> set(doc_ids)
        self.index = {}
        # doc_id -> full document object
        self.docmap = {}

    def __add_document(self, doc_id, text):

        tokens = tokenize_text(text)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        term = term.lower()
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)

    def build(self, movies):
        """
        movies: iterable of movie dicts with keys:
                - 'id'
                - 'title'
                - 'description'
        """
        for m in movies:
            doc_id = m["id"]
            self.docmap[doc_id] = m

            text = f"{m['title']} {m['description']}"
            self.__add_document(doc_id, text)

    def save(self):
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self):
        index_path = "cache/index.pkl"
        docmap_path = "cache/docmap.pkl"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found at {index_path}")

        if not os.path.exists(docmap_path):
            raise FileNotFoundError(f"Docmap file not found at {docmap_path}")

        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)