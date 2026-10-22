"""The Curator Viewer's write token.

The viewer refuses any change to a session made without an API key unless the
change carries the write token that the viewer returned when it made the
session. Before, anyone who had a session's public link could change it.
"""

import asyncio
import itertools
import sqlite3
from types import SimpleNamespace

import pytest
from datasets import Dataset

from bespokelabs.curator import client as client_module
from bespokelabs.curator.agent.agent import Agent, MultiTurnAgents
from bespokelabs.curator.client import WRITE_TOKEN_HEADER, Client
from bespokelabs.curator.constants import BASE_CLIENT_URL
from bespokelabs.curator.db import MetadataDB
from bespokelabs.curator.llm.llm import _open_viewer_session

KEY = "bespoke-" + "ab" * 32
METADATA = {
    "run_hash": "run",
    "dataset_hash": "dataset",
    "prompt_func": "def prompt(row): ...",
    "model_name": "gpt-4o-mini",
    "response_format": "text",
    "batch_mode": False,
    "timestamp": "2026-10-22T00:00:00",
}


class Viewer:
    """The viewer's session routes. It records each request's headers."""

    def __init__(self, tokens=True):
        self.tokens = tokens
        self.created = []
        self.requests = []
        self._ids = itertools.count()

    def post(self, url, json=None, headers=None, timeout=None, **kwargs):
        if not url.startswith(BASE_CLIENT_URL):
            # Another service, e.g. the OpenAI rate-limit probe of an Agent's
            # backend: an empty answer, without the network.
            return SimpleNamespace(status_code=200, headers={}, json=lambda: {}, text="")
        n = next(self._ids)
        self.created.append(headers)
        body = {"session_id": f"session-{n}"}
        if self.tokens and "Authorization" not in headers:
            body["write_token"] = f"token-{n}"
        return SimpleNamespace(status_code=200, json=lambda: body, text="")

    def async_client(self):
        viewer = self

        class AsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def put(self, url, json, headers):
                viewer.requests.append(("PUT", url, headers))
                return SimpleNamespace(status_code=200, text="")

            async def post(self, url, data, headers):
                viewer.requests.append(("POST", url, headers))
                return SimpleNamespace(status_code=200, text="")

            async def aclose(self):
                pass

        return AsyncClient()


@pytest.fixture
def viewer(monkeypatch):
    monkeypatch.setenv("CURATOR_VIEWER", "1")
    monkeypatch.delenv("BESPOKE_API_KEY", raising=False)
    fake = Viewer()
    monkeypatch.setattr(client_module.requests, "post", fake.post)
    monkeypatch.setattr(client_module.httpx, "AsyncClient", fake.async_client)
    return fake


def make_every_change(client):
    """Send every change that a run makes to its session."""
    tracker = SimpleNamespace(cost_info=lambda: {"total_cost_milli_dollars": 1})

    async def changes():
        await client.session_inprogress()
        await client.stream_response('{"parsed_response_message": []}', 0)
        await client.log_cost_projection(tracker, force_log=True)
        await client.session_completed()
        await client.close()

    asyncio.run(changes())


def test_a_session_made_without_a_key_sends_its_write_token_with_every_change(viewer):
    client = Client()
    assert client.create_session(dict(METADATA)) == "session-0"
    assert client.write_token == "token-0"
    make_every_change(client)
    assert len(viewer.requests) == 4
    for _, _, headers in viewer.requests:
        assert headers == {WRITE_TOKEN_HEADER: "token-0"}


def test_a_session_made_with_a_key_sends_the_key_and_no_token(viewer, monkeypatch):
    monkeypatch.setenv("BESPOKE_API_KEY", KEY)
    client = Client()
    client.create_session(dict(METADATA))
    assert client.write_token is None
    make_every_change(client)
    assert len(viewer.requests) == 4
    for _, _, headers in viewer.requests:
        assert headers == {"Authorization": f"Bearer {KEY}"}


def test_a_viewer_without_write_tokens_gets_the_same_requests_as_before(viewer):
    viewer.tokens = False
    client = Client()
    client.create_session(dict(METADATA))
    make_every_change(client)
    assert len(viewer.requests) == 4
    for _, _, headers in viewer.requests:
        assert headers == {}


def test_naming_the_same_session_again_keeps_its_token(viewer):
    client = Client()
    session = client.create_session(dict(METADATA))
    # As MultiTurnAgents does after creating the session.
    client.create_session(dict(METADATA), session_id=session)
    assert client.write_token == "token-0"
    client.create_session(dict(METADATA), session_id="another-session")
    assert client.write_token is None


class Talker(Agent):
    def prompt(self, row):
        return row["prompt"]


def test_a_first_agent_run_uploads_its_rows_with_the_token_and_a_second_run_does_not(viewer, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mocked")
    seeder = Talker(name="seeder", model_name="gpt-4o-mini", backend="openai")
    partner = Talker(name="partner", model_name="gpt-4o-mini", backend="openai")
    agents = MultiTurnAgents(seeder=seeder, partner=partner, max_length=2, seed_message="hello")
    rows = Dataset.from_list([{"role": "seeder", "content": "hello"}, {"role": "partner", "content": "hi"}])

    async def conversation(working_dir):
        # The conversation itself, without a model.
        return rows

    monkeypatch.setattr(agents._processor, "run", conversation)
    monkeypatch.setattr(agents._processor, "status_tracker", None, raising=False)

    result = agents(working_dir=str(tmp_path))
    assert result.viewer_url.endswith("/session-0")
    uploads = [r for r in viewer.requests if r[0] == "POST"]
    assert len(uploads) == 2
    for _, _, headers in viewer.requests:
        assert headers == {WRITE_TOKEN_HEADER: "token-0"}

    # The second run of the same conversation finds it on the viewer.
    agents(working_dir=str(tmp_path))
    assert len([r for r in viewer.requests if r[0] == "POST"]) == 2


def test_the_cache_keeps_write_tokens_beside_the_runs_table(tmp_path):
    db = MetadataDB(str(tmp_path / "metadata.db"))
    # A cache from before write tokens has no token table.
    assert db.get_viewer_write_token("session-0") is None
    db.store_metadata({**METADATA, "session_id": "session-0", "is_hosted_viewer_synced": True})
    db.store_viewer_write_token("session-0", "token-0")
    db.store_viewer_write_token("session-1", None)
    assert db.get_viewer_write_token("session-0") == "token-0"
    assert db.get_viewer_write_token("session-1") is None
    # The runs table is unchanged, so its schema check still passes.
    db.validate_schema()


def run_twice(tmp_path, first_token_kept=True):
    """Run once, then open the session for a second run of the same run hash."""
    db = MetadataDB(str(tmp_path / "metadata.db"))
    first = Client()
    metadata = dict(METADATA)
    session = _open_viewer_session(first, db, metadata, None, False)
    db.store_metadata({**metadata, "session_id": session, "is_hosted_viewer_synced": False})
    db.update_sync_viewer_flag("run", True)
    if not first_token_kept:
        # As an older Curator left the cache.
        with sqlite3.connect(db.db_path) as conn:
            conn.execute("DELETE FROM viewer_write_tokens")
    second = Client()
    existing = db.get_existing_session_id("run")
    reopened = _open_viewer_session(second, db, dict(METADATA), existing, db.check_existing_hosted_sync("run"))
    return db, session, second, reopened


def test_a_second_run_adds_to_the_session_with_the_token_it_kept(viewer, tmp_path):
    db, session, second, reopened = run_twice(tmp_path)
    assert reopened == session
    assert second.write_token == "token-0"
    assert len(viewer.created) == 1


def test_a_second_run_without_the_token_gets_a_new_session(viewer, tmp_path):
    db, session, second, reopened = run_twice(tmp_path, first_token_kept=False)
    assert (session, reopened) == ("session-0", "session-1")
    assert second.write_token == "token-1"
    # The next run adds to the new session, with its token.
    assert db.get_existing_session_id("run") == "session-1"
    assert db.get_viewer_write_token("session-1") == "token-1"


def test_a_second_run_with_a_key_adds_to_the_session_without_a_token(viewer, tmp_path, monkeypatch):
    monkeypatch.setenv("BESPOKE_API_KEY", KEY)
    db, session, second, reopened = run_twice(tmp_path)
    assert reopened == session
    assert second.write_token is None
    assert len(viewer.created) == 1
