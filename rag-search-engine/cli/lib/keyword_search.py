import math
import os
import pickle
import string
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
    BM25_K1,
    BM25_B,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths={}
        self.doc_lengths_path=os.path.join(CACHE_DIR,"doc_lengths.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        # Store document length
        self.doc_lengths[doc_id] = len(tokens)

        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0

        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")

        token = tokens[0]

        N = len(self.docmap)
        df = len(self.index[token])

        # BM25 IDF formula:
        # log((N - df + 0.5) / (df + 0.5) + 1)
        return math.log(((N - df + 0.5) / (df + 0.5)) + 1)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)

        if tf == 0:
            return 0.0

        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()

        if avg_doc_length == 0:
            return 0.0

        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)
    
    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT):
        tokens = tokenize_text(query)

        scores = defaultdict(float)

        # Collect candidate documents (only those containing at least one query term)
        candidate_docs = set()
        for token in tokens:
            candidate_docs.update(self.get_documents(token))

        # Calculate BM25 score for each candidate document
        for doc_id in candidate_docs:
            total_score = 0.0

            for token in tokens:
                # Only compute if term exists in document
                if doc_id in self.index.get(token, set()):
                    total_score += self.bm25(doc_id,token)

            if total_score > 0:
                scores[doc_id] = total_score

        # Sort documents by score (highest first)
        ranked_docs = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top `limit` documents with their scores
        results = []
        for doc_id, score in ranked_docs[:limit]:
            results.append({
                "doc": self.docmap[doc_id],
                "score": score
            })

        return results



def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id,term,k1,b):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id,term,k1,b)

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)
