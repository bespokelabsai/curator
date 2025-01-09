import argparse
import logging

from datasets import Dataset

from bespokelabs.curator import LLM

logger = logging.getLogger("bespokelabs.curator")
logger.setLevel(logging.INFO)


def main(delete_cache: bool = False):
    dataset = Dataset.from_dict({"prompt": ["just say 'hi'"] * 3})

    prompter = LLM(
        prompt_func=lambda row: row["prompt"],
        model_name="gpt-4o-mini",
        response_format=None,
        delete_cache=delete_cache,
    )

    dataset = prompter(dataset)
    print(dataset.to_pandas())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prompter with cache control")
    parser.add_argument(
        "--delete-cache",
        action="store_true",
        help="Delete the cache before running",
    )
    args = parser.parse_args()
    main(delete_cache=args.delete_cache)
