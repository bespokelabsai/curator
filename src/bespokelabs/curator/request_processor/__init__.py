"""Module for processing requests."""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_JSON_COST_PATH = os.path.join(BASE_DIR, "_default_rate_limits.json")

with open(_JSON_COST_PATH, "r") as f:
    _DEFAULT_COST_MAP = json.load(f)
