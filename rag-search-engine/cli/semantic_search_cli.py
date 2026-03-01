#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    SemanticSearch,
)
from lib.search_utils import load_movies


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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()