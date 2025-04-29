"""Models module for Curator."""

import os
import json
import requests
import typing as t
from abc import ABC, abstractmethod
from ..log import logger

from .models import Models

__all__ = ["Models"]