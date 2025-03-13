import os
import uuid
from dataclasses import dataclass
from typing import ClassVar, Optional

import posthog

from bespokelabs.curator.constants import _DEFAULT_CACHE_DIR


def _random_distinct_id():
    # check if cache
    default_cache_dir = os.path.expanduser(_DEFAULT_CACHE_DIR)
    os.makedirs(default_cache_dir, exist_ok=True)
    distinct_id_file = os.path.join(default_cache_dir, ".curator_config")

    if os.path.exists(distinct_id_file):
        with open(distinct_id_file) as f:
            random_distinct_id = uuid.UUID(f.read().strip())
    else:
        random_distinct_id = uuid.uuid4()
        with open(distinct_id_file, "w") as f:
            f.write(str(random_distinct_id))

    return random_distinct_id


@dataclass
class TelemetryEvent:
    """Base class for all telemetry events."""

    event_type: str
    metadata: dict
    distinct_id: ClassVar[str] = _random_distinct_id()


@dataclass
class PosthogConfig:
    """Configuration settings for PostHog client."""

    api_key: str
    enabled: bool = True
    debug: bool = False
    host: Optional[str] = None


class PosthogClient:
    """Client for sending telemetry events to PostHog analytics.

    This uses a write-only project API key that can only create new events.
    It cannot read events or access other PostHog data, making it safe for public apps.
    """

    def __init__(self, config: PosthogConfig):
        """Initialize PostHog client with the given configuration."""
        self.config = config
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Set up the PostHog client with configuration settings."""
        posthog.project_api_key = self.config.api_key
        posthog.debug = self.config.debug
        posthog.disable_geoip = False

        if self.config.host:
            posthog.host = self.config.host

    def capture(self, event: TelemetryEvent) -> None:
        """Capture and send a telemetry event to PostHog.

        Args:
            event: The telemetry event to capture
        """
        if not self.config.enabled:
            return

        try:
            posthog.capture(distinct_id=event.distinct_id, event=event.event_type, properties=event.metadata)
        except Exception:
            pass


# Initialize the telemetry client with environment-based configuration
config = PosthogConfig(
    api_key="phc_HGGTf1LmtsUnBaVBufgIwRsAwdkvH3cSsDKgW5RnJz8",
    enabled=os.getenv("TELEMETRY_ENABLED", "true").lower() in ("true", "1", "t"),
    debug=os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "t"),
    host=os.getenv("POSTHOG_HOST"),
)

telemetry_client = PosthogClient(config=config)
