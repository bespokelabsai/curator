from bespokelabs.curator import LLM
from datasets import Dataset
import logging
import argparse

# python tests/simple_online.py --log-level DEBUG --model claude-3-5-haiku-20241022


def main(args):
    if args.log_level is not None:
        logger = logging.getLogger("bespokelabs.curator")
        logger.setLevel(args.log_level)

    dataset = Dataset.from_dict({"prompt": ["write me a poem"] * args.n_requests})

    prompter = LLM(
        prompt_func=lambda row: row["prompt"],
        model_name=args.model,
        max_requests_per_minute=args.max_requests_per_minute,
        max_tokens_per_minute=args.max_tokens_per_minute,
    )

    dataset = prompter(dataset, batch_cancel=args.cancel)
    print(dataset.to_pandas())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple batch test bed")
    parser.add_argument("--cancel", action="store_true", default=False, help="Cancel the batches")
    parser.add_argument("--n-requests", type=int, help="Number of requests to process", default=3)
    parser.add_argument(
        "--log-level",
        type=lambda x: getattr(logging, x.upper()),
        default=None,
        help="Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument("--model", type=str, help="Model to use", default="gemini/gemini-1.5-flash")
    parser.add_argument(
        "--max-requests-per-minute", type=int, help="Max requests per minute", default=None
    )
    parser.add_argument(
        "--max-tokens-per-minute", type=int, help="Max tokens per minute", default=None
    )
    args = parser.parse_args()
    main(args)
