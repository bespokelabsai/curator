import os

import vcr

from bespokelabs.curator.utils import push_to_viewer


def test_smoke(mock_dataset):
    vcr_path = "tests/integrations/common_fixtures"
    mode = os.environ.get("VCR_MODE")
    vcr_config = vcr.VCR(
        serializer="yaml",
        cassette_library_dir=vcr_path,
        record_mode=mode,
    )
    with vcr_config.use_cassette("viewer.yaml"):
        # HF url
        push_to_viewer("zed-industries/zeta", hf_params={"split": "train[:10]"})

        # HF dataset
        push_to_viewer(mock_dataset)
