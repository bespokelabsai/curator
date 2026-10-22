import asyncio
import json
import os
import time
import typing as t
import uuid

import httpx
import requests

from bespokelabs.curator.constants import BASE_CLIENT_URL, PUBLIC_CURATOR_VIEWER_DATASET_URL
from bespokelabs.curator.log import logger

N_CONCURRENT_VIEWER_REQUESTS = 100
# The viewer gives the creator of a session made without an API key a write
# token, and refuses any change to that session that does not carry it.
WRITE_TOKEN_HEADER = "X-Viewer-Write-Token"


class _SessionStatus:
    """A class to represent the status of a session."""

    STARTED = "STARTED"
    INPROGRESS = "INPROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Client:
    """A class to represent the client for the Curator Viewer."""

    def __init__(self, hosted: bool = False) -> None:
        """Initialize the client."""
        self._session = None
        self._write_token = None
        self._refused = False
        self._state = None
        self._last_cost_projection_time = 0

        # Use environment variable to determine if hosted
        self._hosted = os.environ.get("CURATOR_VIEWER") in ["True", "true", "1", "t"]
        self._hosted = self._hosted or hosted
        self.semaphore = asyncio.Semaphore(N_CONCURRENT_VIEWER_REQUESTS)
        self._async_client = None
        self.api_key = os.environ.get("BESPOKE_API_KEY")
        if self.api_key:
            logger.info("Authenticated with Curator API key")
        self._headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    @property
    def session(self):
        """Get the session ID."""
        return self._session

    @property
    def hosted(self):
        """Check if the client is hosted."""
        return self._hosted

    @property
    def curator_viewer_url(self):
        """Get the curator viewer URL."""
        return f"{PUBLIC_CURATOR_VIEWER_DATASET_URL}/{self.session}" if self.session else None

    @property
    def write_token(self):
        """The token that lets this client change a session it made without an API key."""
        return self._write_token

    def can_write_to(self, write_token: str | None) -> bool:
        """Whether this client may change an existing session whose stored write token is write_token.

        A session made with an API key changes with that key. One made without
        a key changes only with the write token that the viewer gave its creator.
        """
        return bool(self.api_key or write_token)

    @property
    def refused(self) -> bool:
        """Whether the viewer refused a change to this client's session. The client then sends it no more."""
        return self._refused

    def forget_session(self) -> None:
        """Drop this client's session, so that the next create_session makes a new one."""
        self._session = None
        self._write_token = None
        self._refused = False
        self._state = None

    def _write_headers(self) -> t.Dict[str, str]:
        """The headers of a request that changes the session."""
        if self._write_token:
            return {**self._headers, WRITE_TOKEN_HEADER: self._write_token}
        return self._headers

    def _check_write(self, response, action: str) -> None:
        """Note a change to the session that the viewer refused, or log any other failure."""
        if response.status_code in (401, 403):
            # Every later change would be refused too, so say so once.
            if not self._refused:
                logger.warning(
                    f"The Curator Viewer refused a change to session {self.session} ({response.status_code}), "
                    "for example because an older Curator made the session without an API key. "
                    "No more changes are sent to it."
                )
            self._refused = True
        elif response.status_code != 200:
            logger.debug(f"Failed to {action}: {response.status_code}, {response.text}")

    def create_session(self, metadata: t.Dict, session_id: str | None = None, write_token: str | None = None) -> str | None:
        """Sends a POST request to the server to create a session.

        Args:
            metadata: The session's metadata.
            session_id: An existing session to use instead of a new one.
            write_token: The write token that the viewer gave when it made session_id, for a session made without an API key.
        """
        if not self.hosted:
            return str(uuid.uuid4().hex)

        if session_id:
            # Naming the session this client already has keeps its token.
            if write_token is not None or session_id != self._session:
                self._write_token = write_token
            self._session = session_id
            return session_id
        if self.session:
            return self.session
        metadata.update({"status": _SessionStatus.STARTED})

        response = requests.post(f"{BASE_CLIENT_URL}/sessions", json=metadata, headers=self._headers, timeout=30)

        if response.status_code == 200:
            self._session = response.json().get("session_id")
            # Without an API key the session has no owner, and only this token
            # can change it. An older viewer sends none.
            self._write_token = response.json().get("write_token")
            self._state = _SessionStatus.STARTED
            return self.session
        else:
            logger.warning(f"Failed to create session: {response.status_code}, {response.text}")
            return str(uuid.uuid4().hex)

    async def _update_state(self):
        if self._refused:
            return
        async with httpx.AsyncClient() as client:
            response = await client.put(f"{BASE_CLIENT_URL}/sessions/{self.session}", json={"status": self._state}, headers=self._write_headers())
        self._check_write(response, "update session status")

    async def session_inprogress(self):
        """Updates the session status to inprogress."""
        self._state = _SessionStatus.INPROGRESS
        if not self._hosted and not self.session:
            return
        await self._update_state()

    async def session_completed(self):
        """Updates the session status to completed."""
        self._state = _SessionStatus.COMPLETED
        if not self._hosted and not self.session:
            return
        await self._update_state()

    async def session_failed(self):
        """Updates the session status to failed."""
        self._state = _SessionStatus.FAILED
        if not self._hosted and not self.session:
            return
        await self._update_state()

    async def stream_response(self, response_data: str, idx: int):
        """Streams the response data to the server."""
        if not self._hosted and not self.session:
            return
        if self._refused:
            return
        if self._async_client is None:
            self._async_client = httpx.AsyncClient()

        response_data = json.dumps({"response_data": response_data})
        async with self.semaphore:
            # A row that waited here while another was refused sends nothing.
            if self._refused:
                return
            response = await self._async_client.post(
                f"{BASE_CLIENT_URL}/sessions/{self.session}/responses/{idx}", data=response_data, headers=self._write_headers()
            )

        self._check_write(response, "stream response to curator Viewer")

    async def close(self):
        """Close the client."""
        if self._async_client:
            await self._async_client.aclose()

    async def log_cost_projection(self, status_tracker, force_log: bool = False):
        """Log the cost projection to the server."""
        if not self._hosted and not self.session:
            return
        if self._refused:
            return
        # Track the last time we sent a cost projection
        current_time = time.time()

        if (current_time - self._last_cost_projection_time) < 10 and not force_log:
            return

        if self._async_client is None:
            self._async_client = httpx.AsyncClient()

        info = status_tracker.cost_info()

        async with self.semaphore:
            if self._refused:
                return
            response = await self._async_client.put(f"{BASE_CLIENT_URL}/sessions/{self.session}", json=info, headers=self._write_headers())

        self._check_write(response, "log cost projection")

        # Update the last projection time
        self._last_cost_projection_time = current_time
