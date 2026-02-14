<!-- Implementing a simple RAG system for a Netflix kind of streaming service -->

Stages :

1.Preprocessing
    -Case Insensitivity
    -Remove Punctuation
    -Tokenization
    -Stop Words
    -Stemming

2.TF -IDF(Term frequency and Inverted Document frequency)
    -An inverted index stores a mapping from each word to the list of movie IDs that contain that word.
    -Tokenizes the query
    -Looks up each word in the inverted index
    -Gets matching movie IDs instantly
    -Uses docmap to fetch movie titles
    -Stops after 5 results
