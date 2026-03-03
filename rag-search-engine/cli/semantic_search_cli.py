#!/usr/bin/env python3

import argparse,re
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    SemanticSearch,
)
from lib.search_utils import load_movies
from lib.semantic_search import ChunkedSemanticSearch

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the model")

    # Embed text
    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_text_parser.add_argument("text", type=str, help="Single string argument")

    # Verify embeddings
    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    # Embed query
    query_embeddings_parser = subparsers.add_parser(
        "embedquery", help="Convert query into embedding"
    )
    query_embeddings_parser.add_argument("query", type=str, help="Query string")

    # 🔎 SEARCH COMMAND
    search_parser = subparsers.add_parser(
        "search", help="Search movies semantically"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )

 
    chunk_parser = subparsers.add_parser("chunk", help="Chunk the text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk (default: 200)",
    )

    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of overlapping words between chunks (default: 0)",
    )

        #SEMANTIC CHUNK COMMAND
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk text by sentence boundaries"
    )
    semantic_chunk_parser.add_argument(
        "text",
        type=str,
        help="Text to semantically chunk",
    )
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum sentences per chunk (default: 4)",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of overlapping sentences (default: 0)",
    )

    embed_chunks_parser = subparsers.add_parser(
    "embed_chunks",
    help="Generate chunked embeddings for movie descriptions")



    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case "search":
            # 1️⃣ Create search instance
            ss = SemanticSearch()

            # 2️⃣ Load documents
            documents = load_movies()

            # 3️⃣ Load or build embeddings
            ss.load_or_create_embeddings(documents)

            # 4️⃣ Perform search
            results = ss.search(args.query, args.limit)

            # 5️⃣ Print results
            print(f"\nTop {len(results)} results for: '{args.query}'\n")

            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result['title']}")
                print(f"   Score: {result['score']:.4f}")
                print(f"   {result['description']}\n")


        case "chunk":
            text = args.text
            chunk_size = args.chunk_size
            overlap = args.overlap

            if overlap >= chunk_size:
                print("Error: overlap must be smaller than chunk size")
                return

            words = text.split()
            chunks = []

            step = chunk_size - overlap if overlap > 0 else chunk_size

            for i in range(0, len(words), step):
                chunk_words = words[i:i + chunk_size]

                if not chunk_words:
                    break

                chunk = " ".join(chunk_words)
                chunks.append(chunk)

                if i + chunk_size >= len(words):
                    break

            print(f"Chunking {len(text)} characters")

            for idx, chunk in enumerate(chunks, start=1):
                print(f"{idx}. {chunk}")
        
        case "semantic_chunk":
            text = args.text
            max_chunk_size = args.max_chunk_size
            overlap = args.overlap

            if overlap >= max_chunk_size:
                print("Error: overlap must be smaller than max chunk size")
                return

            # Split into sentences using regex
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())

            chunks = []

            step = max_chunk_size - overlap if overlap > 0 else max_chunk_size

            for i in range(0, len(sentences), step):
                chunk_sentences = sentences[i:i + max_chunk_size]

                if not chunk_sentences:
                    break

                chunk = " ".join(chunk_sentences)
                chunks.append(chunk)

                if i + max_chunk_size >= len(sentences):
                    break

            print(f"Semantically chunking {len(text)} characters")

            for idx, chunk in enumerate(chunks, start=1):
                print(f"{idx}. {chunk}")
        
        case "embed_chunks":
            documents = load_movies()

            css = ChunkedSemanticSearch()

            embeddings = css.load_or_create_chunk_embeddings(documents)

            print(f"Generated {len(embeddings)} chunked embeddings")
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()