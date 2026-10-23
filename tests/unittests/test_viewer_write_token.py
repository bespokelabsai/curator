"""The Curator Viewer's write token.

The viewer refuses any change to a session made without an API key unless the
change carries the write token that the viewer returned when it made the
session. Before, anyone who had a session's public link could change it.
"""

import asyncio
import itertools
import logging
import os
import sqlite3
from types import SimpleNamespace

import pytest
from datasets import Dataset

from bespokelabs import curator
from bespokelabs.curator import client as client_module
from bespokelabs.curator import db as db_module
from bespokelabs.curator.agent.agent import Agent, MultiTurnAgents
from bespokelabs.curator.client import WRITE_TOKEN_HEADER, Client
from bespokelabs.curator.constants import BASE_CLIENT_URL
from bespokelabs.curator.db import MetadataDB
from bespokelabs.curator.llm.llm import _open_viewer_session
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.status_tracker import agent_status_tracker

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
    """The viewer's session routes. It records each request's headers.

    It refuses every change to a session in `refuse`, as the viewer does to a
    session whose owner the request cannot show it is. While `creates` is
    False, it fails to make sessions.
    """

    def __init__(self, tokens=True):
        self.tokens = tokens
        self.refuse = set()
        self.creates = True
        self.created = []
        self.requests = []
        self._ids = itertools.count()

    def answer(self, url):
        session = url[len(f"{BASE_CLIENT_URL}/sessions/") :].split("/")[0]
        return SimpleNamespace(status_code=403 if session in self.refuse else 200, text="")

    def sent_to(self, session):
        return [r for r in self.requests if r[1].startswith(f"{BASE_CLIENT_URL}/sessions/{session}")]

    def post(self, url, json=None, headers=None, timeout=None, **kwargs):
        if not url.startswith(BASE_CLIENT_URL):
            # Another service, e.g. the OpenAI rate-limit probe of an Agent's
            # backend: an empty answer, without the network.
            return SimpleNamespace(status_code=200, headers={}, json=lambda: {}, text="")
        if not self.creates:
            return SimpleNamespace(status_code=503, json=lambda: {}, text="unavailable")
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

            # Each answer takes a turn of the event loop, as a real request
            # does, so concurrent requests interleave.
            async def put(self, url, json, headers):
                viewer.requests.append(("PUT", url, headers))
                await asyncio.sleep(0)
                return viewer.answer(url)

            async def post(self, url, data, headers):
                viewer.requests.append(("POST", url, headers))
                await asyncio.sleep(0)
                return viewer.answer(url)

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


def conversation_of_two(monkeypatch):
    """Two agents whose conversation runs without a model and yields two rows."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mocked")
    # The tracker starts rich's live display when it is made, and only the
    # conversation stops it. Rich allows one at a time, so use plain progress.
    monkeypatch.setattr(agent_status_tracker, "USE_RICH_DISPLAY", False)
    seeder = Talker(name="seeder", model_name="gpt-4o-mini", backend="openai")
    partner = Talker(name="partner", model_name="gpt-4o-mini", backend="openai")
    agents = MultiTurnAgents(seeder=seeder, partner=partner, max_length=2, seed_message="hello")
    rows = Dataset.from_list([{"role": "seeder", "content": "hello"}, {"role": "partner", "content": "hi"}])

    async def conversation(working_dir):
        return rows

    monkeypatch.setattr(agents._processor, "run", conversation)
    monkeypatch.setattr(agents._processor, "status_tracker", None, raising=False)
    return agents


def test_a_first_agent_run_uploads_its_rows_with_the_token_and_a_second_run_does_not(viewer, tmp_path, monkeypatch):
    agents = conversation_of_two(monkeypatch)

    result = agents(working_dir=str(tmp_path))
    assert result.viewer_url.endswith("/session-0")
    uploads = [r for r in viewer.requests if r[0] == "POST"]
    assert len(uploads) == 2
    for _, _, headers in viewer.requests:
        assert headers == {WRITE_TOKEN_HEADER: "token-0"}

    # The second run of the same conversation finds it on the viewer.
    agents(working_dir=str(tmp_path))
    assert len([r for r in viewer.requests if r[0] == "POST"]) == 2


def test_an_agent_run_after_one_without_the_viewer_pushes_once_and_keeps_its_session(viewer, tmp_path, monkeypatch):
    agents = conversation_of_two(monkeypatch)
    monkeypatch.delenv("CURATOR_VIEWER")
    agents(working_dir=str(tmp_path))  # caches a session ID the viewer never saw
    assert viewer.created == []

    monkeypatch.setenv("CURATOR_VIEWER", "1")
    assert agents(working_dir=str(tmp_path)).viewer_url.endswith("/session-0")
    assert len([r for r in viewer.requests if r[0] == "POST"]) == 2
    # The next run uses that session, not the ID cached by the first run.
    assert agents(working_dir=str(tmp_path)).viewer_url.endswith("/session-0")
    assert len([r for r in viewer.requests if r[0] == "POST"]) == 2
    assert len(viewer.created) == 1


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
    # The new session holds none of the run's rows yet. Until the run's
    # upload completes, the run is not synced, so an upload that stops
    # halfway is pushed again by the next run.
    assert db.check_existing_hosted_sync("run") is False


def test_a_run_whose_upload_did_not_finish_adds_to_the_same_session(viewer, tmp_path):
    db = MetadataDB(str(tmp_path / "metadata.db"))
    first = Client()
    session = _open_viewer_session(first, db, dict(METADATA), None, False)
    # The upload stopped before the run was marked synced.
    db.store_metadata({**METADATA, "session_id": session, "is_hosted_viewer_synced": False})
    second = Client()
    reopened = _open_viewer_session(second, db, dict(METADATA), db.get_existing_session_id("run"), db.check_existing_hosted_sync("run"))
    assert reopened == session
    assert second.write_token == "token-0"
    assert len(viewer.created) == 1


def test_a_second_run_with_a_key_adds_to_the_session_without_a_token(viewer, tmp_path, monkeypatch):
    monkeypatch.setenv("BESPOKE_API_KEY", KEY)
    db, session, second, reopened = run_twice(tmp_path)
    assert reopened == session
    assert second.write_token is None
    assert len(viewer.created) == 1


def client_warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "refused" in r.getMessage()]


def test_a_refused_change_is_reported_once_and_no_more_changes_are_sent(viewer, monkeypatch, caplog):
    monkeypatch.setenv("BESPOKE_API_KEY", KEY)
    client = Client()
    # A session that an older Curator made without a key: it has no owner.
    client.create_session(dict(METADATA), session_id="legacy")
    viewer.refuse.add("legacy")
    with caplog.at_level(logging.WARNING):
        make_every_change(client)
    assert client.refused
    assert len(viewer.requests) == 1
    assert len(client_warnings(caplog)) == 1


def test_rows_waiting_to_send_when_a_change_is_refused_send_nothing(viewer, monkeypatch):
    monkeypatch.setenv("BESPOKE_API_KEY", KEY)
    client = Client()
    client.create_session(dict(METADATA), session_id="legacy")
    viewer.refuse.add("legacy")
    # One request at a time, so the other rows wait while the first is refused.
    client.semaphore = asyncio.Semaphore(1)

    async def send_rows():
        await asyncio.gather(*(client.stream_response('{"parsed_response_message": []}', idx) for idx in range(3)))
        await client.close()

    asyncio.run(send_rows())
    assert len(viewer.requests) == 1


class Echo(curator.LLM):
    def prompt(self, row):
        return row["text"]


def refused_rerun(viewer, tmp_path, monkeypatch):
    """Run once as an older Curator did: without a key, and given no token.

    Then set a key. The session has no owner, so the viewer refuses every change to it.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mocked")
    llm = Echo(model_name="gpt-4o-mini", backend="openai")

    def run(dataset, working_dir, parse_func_hash, prompt_formatter):
        # Report progress to the run's session, as the request processors do.
        client = llm._request_processor.viewer_client

        async def report():
            await client.session_inprogress()
            await client.stream_response('{"parsed_response_message": []}', 0)
            await client.session_completed()

        run_in_event_loop(report())
        return dataset

    monkeypatch.setattr(llm._request_processor, "run", run)
    monkeypatch.setattr(llm._request_processor, "_is_cached_dataset", False, raising=False)
    monkeypatch.setattr(llm._request_processor, "tracker", None, raising=False)

    rows = Dataset.from_list([{"text": "a"}, {"text": "b"}])
    viewer.tokens = False
    assert llm(rows, working_dir=str(tmp_path)).viewer_url.endswith("/session-0")
    monkeypatch.setenv("BESPOKE_API_KEY", KEY)
    viewer.refuse.add("session-0")
    return llm, rows


def rows_sent_to(viewer, session):
    """The indices of the rows posted to a session, sorted, with any repeats."""
    return sorted(int(r[1].rsplit("/", 1)[1]) for r in viewer.sent_to(session) if r[0] == "POST")


def cached_runs(tmp_path):
    with sqlite3.connect(tmp_path / "metadata.db") as conn:
        return conn.execute("SELECT session_id, is_hosted_viewer_synced FROM runs").fetchall()


def test_a_run_whose_session_the_viewer_refuses_pushes_its_rows_to_a_new_session(viewer, tmp_path, monkeypatch, caplog):
    llm, rows = refused_rerun(viewer, tmp_path, monkeypatch)
    sent_before = len(viewer.sent_to("session-0"))
    with caplog.at_level(logging.WARNING):
        response = llm(rows, working_dir=str(tmp_path))

    # One refused change, then nothing more to that session.
    assert len(viewer.sent_to("session-0")) == sent_before + 1
    # A new session, made with the key, gets every row once.
    assert len(viewer.created) == 2
    assert viewer.created[1] == {"Authorization": f"Bearer {KEY}"}
    assert rows_sent_to(viewer, "session-1") == list(range(len(rows)))
    assert response.viewer_url.endswith("/session-1")
    assert len(client_warnings(caplog)) == 2  # the client's, and the run's
    # The next run adds to the new session.
    assert cached_runs(tmp_path) == [("session-1", 1)]


def test_a_run_that_gets_no_new_session_caches_none_and_the_next_run_pushes(viewer, tmp_path, monkeypatch, caplog):
    llm, rows = refused_rerun(viewer, tmp_path, monkeypatch)
    viewer.creates = False
    with caplog.at_level(logging.WARNING):
        llm(rows, working_dir=str(tmp_path))
    # No made-up session ID is cached, and the run is not synced.
    assert cached_runs(tmp_path) == [("session-0", 0)]
    assert any("could not make a new session" in message for message in client_warnings(caplog))

    # Once the viewer makes sessions again, the next run pushes every row to a new one.
    viewer.creates = True
    assert llm(rows, working_dir=str(tmp_path)).viewer_url.endswith("/session-1")
    # The run streams row 0 itself; the full push then sends every row once.
    assert rows_sent_to(viewer, "session-1") == sorted([0, *range(len(rows))])
    assert cached_runs(tmp_path) == [("session-1", 1)]


@pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")
def test_the_cache_that_holds_write_tokens_is_readable_only_by_its_owner(tmp_path):
    db = MetadataDB(str(tmp_path / "metadata.db"))
    db.store_metadata({**METADATA, "session_id": "session-0", "is_hosted_viewer_synced": True})
    db.store_viewer_write_token("session-0", "token-0")
    assert os.stat(db.db_path).st_mode & 0o777 == 0o600


def test_no_token_is_kept_when_the_cache_cannot_be_made_owner_only(tmp_path, monkeypatch, caplog):
    db = MetadataDB(str(tmp_path / "metadata.db"))
    db.store_metadata({**METADATA, "session_id": "session-0", "is_hosted_viewer_synced": True})

    def chmod(path, mode):
        # As for a cache file that another user owns.
        raise PermissionError(1, "Operation not permitted", path)

    monkeypatch.setattr(db_module.os, "chmod", chmod)
    with caplog.at_level(logging.WARNING):
        db.store_viewer_write_token("session-0", "token-0")
    assert db.get_viewer_write_token("session-0") is None
    assert any("Not keeping the Curator Viewer write token" in r.getMessage() for r in caplog.records)
