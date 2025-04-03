import hashlib

import pytest
from datasets import Dataset

from bespokelabs.curator.blocks.raft import Raft
from bespokelabs.curator.blocks.simplestrat import StratifiedGenerator

##############################
# Online                     #
##############################


def _hash_string(input_string):
    return hashlib.sha256(input_string.encode("utf-8")).hexdigest()


_ONLINE_BACKENDS = [{"integration": backend} for backend in {"openai"}]


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_basic_raft(temp_working_dir):
    temp_working_dir, backend, vcr_config = temp_working_dir
    hash_book = {
        "openai": "93911791271bde9993f0112f27087bef9e612bb06bb448219dec2ea109657199",
    }

    with vcr_config.use_cassette("basic_block_raft.yaml"):
        with open("tests/integrations/common_fixtures/raft.txt", "rb") as file:
            text = file.read().decode("utf-8")
        raft = Raft(model="gpt-4o-mini", distractors=2, n_questions=1, chunk_size=1024, p=0.95)
        dataset = raft(text)

        qas = [qa[0] for qa in dataset.to_pandas().values.tolist()]
        qas.sort()
        qas = "".join(qas)
        assert _hash_string(qas) == hash_book[backend]


@pytest.mark.parametrize("temp_working_dir", (_ONLINE_BACKENDS), indirect=True)
def test_basic_simplestrat(temp_working_dir):
    temp_working_dir, backend, vcr_config = temp_working_dir
    hash_book = {
        "openai": "110dc28ab2b8bce73dd42a56a2daaaa0bee25632e0dc430577b048653baeec5e",
    }

    with vcr_config.use_cassette("basic_block_simplestrat.yaml"):
        questions = Dataset.from_dict({"question": [f"{i}. Name a periodic element" for i in range(10)]})
        generate = StratifiedGenerator(model_name="gpt-4o-mini")
        qas = generate(questions, working_dir=temp_working_dir)
        qas = [qa[0] for qa in qas.to_pandas().values.tolist()]
        qas.sort()
        qas = "".join(qas)
        assert _hash_string(qas) == hash_book[backend]
