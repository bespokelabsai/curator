import hashlib

import pytest

from bespokelabs.curator.blocks.raft import Raft

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
        "openai": "9906685f6b535a2242529453ad992bc7313e410f173b2311197e466cfce1144f",
    }

    with vcr_config.use_cassette("basic_block_raft.yaml"):
        # Capture the output to verify status tracker
        with open("tests/integrations/common_fixtures/raft.txt", "rb") as file:
            text = file.read().decode("utf-8")
        raft = Raft(model="gpt-4o-mini", distractors=2, n_questions=1, chunk_size=1024, p=0.95)
        dataset = raft(text)

        recipes = "".join([qa[0] for qa in dataset.to_pandas().values.tolist()])
        assert _hash_string(recipes) == hash_book[backend]
