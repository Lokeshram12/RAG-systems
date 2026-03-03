from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from .search_utils import (
    load_movies,
)
import re

class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if text == "":
            raise ValueError("Text is empty")
        return self.model.encode(text)

    def build_embeddings(self, documents):
        """
        Build embeddings from scratch and save them to disk.
        """
        self.documents = documents
        self.document_map = {}

        movies_list = []

        for doc in documents:
            self.document_map[doc["id"]] = doc
            value = f"{doc['title']}: {doc['description']}"
            movies_list.append(value)

        print("Building embeddings (this may take a while)...")

        # Encode list with progress bar
        self.embeddings = self.model.encode(
            movies_list,
            show_progress_bar=True
        )

        # Save to disk
        os.makedirs("cache", exist_ok=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        """
        Load cached embeddings if valid.
        Otherwise rebuild them.
        """
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        cache_path = "cache/movie_embeddings.npy"

        if os.path.exists(cache_path):
            print("Loading embeddings from cache...")
            self.embeddings = np.load(cache_path)

            # Validate length
            if len(self.embeddings) == len(documents):
                return self.embeddings
            else:
                print("Cache size mismatch. Rebuilding embeddings...")

        # If cache missing or invalid → rebuild
        return self.build_embeddings(documents)

    def search(self, query, limit):
        """
        Search for the most similar documents to the query.
        """

        # 1️⃣ Ensure embeddings are loaded
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        # 2️⃣ Generate query embedding
        query_embedding = self.generate_embedding(query)

        results = []

        # 3️⃣ Compute cosine similarity with each document embedding
        for idx, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            document = self.documents[idx]
            results.append((score, document))

        # 4️⃣ Sort by similarity (highest first)
        results.sort(key=lambda x: x[0], reverse=True)

        # 5️⃣ Return top results (up to limit)
        top_results = []

        for score, document in results[:limit]:
            top_results.append({
                "score": float(score),
                "title": document["title"],
                "description": document["description"],
            })

        return top_results


class ChunkedSemanticSearch(SemanticSearch):

    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        """
        Build chunk embeddings from scratch and save them to disk.
        """

        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        all_chunks = []
        chunk_metadata = []

        for movie_idx, doc in enumerate(self.documents):
            description = doc.get("description", "")

            if not description.strip():
                continue

            chunks = semantic_chunk_text(
                description,
                max_chunk_size=4,
                overlap=1
            )

            total_chunks = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)

                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                })

        print("Building chunk embeddings (this may take a while)...")

        self.chunk_embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=True
        )

        self.chunk_metadata = chunk_metadata

        # Save cache
        os.makedirs("cache", exist_ok=True)

        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w") as f:
            json.dump(
                {
                    "chunks": chunk_metadata,
                    "total_chunks": len(all_chunks),
                },
                f,
                indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents):
        """
        Load cached chunk embeddings if available.
        Otherwise rebuild them.
        """

        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        embeddings_path = "cache/chunk_embeddings.npy"
        metadata_path = "cache/chunk_metadata.json"

        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            print("Loading chunk embeddings from cache...")

            self.chunk_embeddings = np.load(embeddings_path)

            with open(metadata_path, "r") as f:
                metadata_json = json.load(f)

            self.chunk_metadata = metadata_json["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)
# ----------------------------
# Top-Level Functions
# ----------------------------

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    ss = SemanticSearch()

    # Load movie documents
    documents = load_movies()

    embeddings = ss.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_chunk_text(text, max_chunk_size=4, overlap=1):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    if overlap >= max_chunk_size:
        raise ValueError("overlap must be smaller than max_chunk_size")

    chunks = []
    step = max_chunk_size - overlap

    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i:i + max_chunk_size]
        if not chunk_sentences:
            break

        chunk = " ".join(chunk_sentences)
        chunks.append(chunk)

        if i + max_chunk_size >= len(sentences):
            break

    return chunks