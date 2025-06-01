import os

import vcr

from bespokelabs.curator.utils import load_dataset, push_to_viewer


def test_smoke(mock_dataset):
    vcr_path = "tests/integrations/common_fixtures"
    os.environ["CURATOR_VIEWER"] = "true"
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


def test_fetch():
    vcr_path = "tests/integrations/common_fixtures"
    os.environ["CURATOR_VIEWER"] = "true"
    mode = os.environ.get("VCR_MODE")
    vcr_config = vcr.VCR(
        serializer="yaml",
        cassette_library_dir=vcr_path,
        record_mode=mode,
    )
    with vcr_config.use_cassette("viewer_fetch.yaml"):
        # HF url
        url = push_to_viewer("zed-industries/zeta", hf_params={"split": "train[:10]"})

        # HF dataset
        dataset_id = url.split("/")[-1]
        ds = load_dataset(dataset_id)
        assert "events" in ds.column_names
        assert len(ds) == 5
