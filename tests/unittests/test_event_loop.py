import asyncio

import pytest

from bespokelabs.curator.request_processor.event_loop import run_in_event_loop


async def _fail(calls):
    calls.append(1)
    raise RuntimeError("original error")


async def _answer():
    return 42


def test_coroutine_error_is_raised_without_a_running_loop():
    calls = []

    with pytest.raises(RuntimeError, match="original error") as exc_info:
        run_in_event_loop(_fail(calls))

    assert calls == [1]
    # Not chained to the "no running event loop" RuntimeError used to detect the loop.
    assert exc_info.value.__context__ is None


def test_coroutine_error_is_raised_inside_a_running_loop():
    """In Jupyter or Colab an event loop is already running when curator is called."""
    calls = []

    async def notebook_cell():
        with pytest.raises(RuntimeError, match="original error"):
            run_in_event_loop(_fail(calls))

    asyncio.run(notebook_cell())

    assert calls == [1]


def test_result_is_returned_inside_a_running_loop():
    async def notebook_cell():
        return run_in_event_loop(_answer())

    assert asyncio.run(notebook_cell()) == 42
