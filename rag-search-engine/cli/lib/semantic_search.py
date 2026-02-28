from sentence_transformers import SentenceTransformer

class SemanticSearch:

    def __init__(self):
        # Load the model (downloads automatically the first time)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model=model

def verify_model():

    ss=SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


