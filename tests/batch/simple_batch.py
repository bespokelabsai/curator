import argparse
import logging

from datasets import Dataset

from bespokelabs.curator import LLM

# python tests/batch/simple_batch.py --log-level DEBUG --n-requests 3 --batch-size 1 --batch-check-interval 10 --model gpt-4o-mini # type: ignore
# python tests/batch/simple_batch.py --log-level DEBUG --n-requests 3 --batch-size 1 --batch-check-interval 10 --model claude-3-5-haiku-20241022 # type: ignore


def main(args):
    if args.log_level is not None:
        logger = logging.getLogger("bespokelabs.curator")
        logger.setLevel(args.log_level)

    dataset = Dataset.from_dict({"prompt": ["just say 'hi'"] * args.n_requests})

    prompter = LLM(
        prompt_func=lambda row: row["prompt"],
        model_name=args.model,
        response_format=None,
        batch=True,
        batch_size=args.batch_size,
        batch_check_interval=args.batch_check_interval,
        base_url=args.base_url,
    )

    dataset = prompter(dataset, batch_cancel=args.cancel)
    print(dataset.to_pandas())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple batch test bed")
    parser.add_argument("--cancel", action="store_true", default=False, help="Cancel the batches")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--batch-size", type=int, default=1_000, help="Batch size")
    parser.add_argument("--batch-check-interval", type=int, default=60, help="Batch check interval")
    parser.add_argument("--n-requests", type=int, help="Number of requests to process")
    parser.add_argument(
        "--log-level",
        type=lambda x: getattr(logging, x.upper()),
        default=None,
        help="Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument("--base-url", type=str, help="Base URL", default=None)
    args = parser.parse_args()
    main(args)
