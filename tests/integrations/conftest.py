import os
import shutil

import pytest
import vcr
from datasets import Dataset

mode = os.environ.get("VCR_MODE", None)
_KEY_MAP = {"openai": "OPENAI_API_KEY", "litellm": "OPENAI_API_KEY"}


@pytest.fixture
def temp_working_dir(request):
    backend = request.param["integration"]
    if mode is None:
        os.environ[_KEY_MAP[backend]] = "sk-mocked-**"
    os.environ["HF_DATASETS_CACHE"] = "/dev/null"
    temp_working_dir = f"tests/integrations/{backend}/fixtures/.test_cache"
    vcr_config = vcr.VCR(
        serializer="yaml",
        cassette_library_dir=f"tests/integrations/{backend}/fixtures",
        record_mode=mode,
    )
    os.makedirs(temp_working_dir, exist_ok=True)

    try:
        yield temp_working_dir, backend, vcr_config
    finally:
        shutil.rmtree(temp_working_dir)


@pytest.fixture
def mock_dataset():
    dataset = Dataset.from_parquet("tests/integrations/common_fixtures/dataset.parquet")
    try:
        yield dataset
    finally:
        # TODO: cleanup?
        pass


@pytest.fixture
def camel_gt_dataset():
    dataset = Dataset.from_parquet("tests/integrations/common_fixtures/camel_gt_dataset.parquet")
    yield dataset
