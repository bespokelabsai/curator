"""Defaults used across curator."""

BATCH_REQUEST_ID_TAG = "custom_id"
_CURATOR_DEFAULT_CACHE_DIR = "~/.cache/curator"
_DEFAULT_CACHE_DIR = "~/.cache"
BASE_CLIENT_URL = "https://api.bespokelabs.ai/v0/viewer"
PUBLIC_CURATOR_VIEWER_HOME_URL = "https://curator.bespokelabs.ai"
PUBLIC_CURATOR_VIEWER_DATASET_URL = PUBLIC_CURATOR_VIEWER_HOME_URL + "/datasets"
_INTERNAL_PROMPT_KEY = "__internal_prompt"
_CACHE_MSG = (
    "If you want to regenerate the dataset, disable or delete the cache.\n See "
    "https://docs.bespokelabs.ai/bespoke-curator/tutorials/automatic-recovery-and-caching#disable-caching "
    "for more information."
)
