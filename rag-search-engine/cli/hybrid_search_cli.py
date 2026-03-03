import argparse
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Normalize command
    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize a list of scores using min-max normalization",
    )
    normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="+",
        help="List of scores to normalize",
    )

    weighted_parser = subparsers.add_parser(
    "weighted-search",
    help="Run hybrid weighted search",
)
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for semantic score (default: 0.5)",
    )
    weighted_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores

            # If no scores are given, don't print anything
            if not scores:
                return

            min_score = min(scores)
            max_score = max(scores)

            # If all scores are the same → print 1.0 for each
            if min_score == max_score:
                for _ in scores:
                    print(f"* {1.0:.4f}")
                return

            # Min-max normalization
            for score in scores:
                normalized = (score - min_score) / (max_score - min_score)
                print(f"* {normalized:.4f}")

        case "weighted-search":
    
            documents = load_movies()

            hs = HybridSearch(documents)

            results = hs.weighted_search(
                query=args.query,
                alpha=args.alpha,
                limit=args.limit,
            )

            top_results = results[:args.limit]

            for idx, result in enumerate(top_results, start=1):
                
                print(f"{idx}. {result['title']}")
                print(f"   Hybrid Score: {result['hybrid']:.3f}")
                print(f"   BM25: {result['bm25']:.3f}, Semantic: {result['semantic']:.3f}")
                print(f"   {result['description'][:100]}...\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()