import string

from nltk.stem import PorterStemmer

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords
from .inverted_index import InvertedIndex

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()

    # Try loading the cached index
    try:
        idx.load()
    except FileNotFoundError:
        print("Error: Inverted index not found. Please run `build` first.")
        return []

    results = []
    seen_doc_ids = set()

    query_tokens = tokenize_text(query)

    for token in query_tokens:
        doc_ids = idx.get_documents(token)

        for doc_id in doc_ids:
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                results.append(idx.docmap[doc_id])

                if len(results) >= limit:
                    break

        if len(results) >= limit:
            break

    # Print results
    for movie in results:
        print(f"{movie['id']} - {movie['title']}")

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


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

