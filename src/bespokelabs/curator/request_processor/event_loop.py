import asyncio

import nest_asyncio


def run_in_event_loop(coroutine):
    """Run a coroutine in the current event loop or create a new one if there isn't one."""
    # First, clean up any existing Rich live displays
    # This prevents "Only one live display may be active at once" errors
    # especially after keyboard interrupts in environments like Colab
    try:
        # Get all live displays from all console instances
        from rich import get_console

        console = get_console()
        if hasattr(console, "_live") and console._live is not None:
            try:
                console._live.stop()
                console._live = None
            except Exception:
                # If stopping fails, just set to None
                console._live = None
    except Exception:
        # If any error occurs during cleanup, just continue
        pass

    try:
        # This call will raise an RuntimError if there is no event loop running.
        asyncio.get_running_loop()

        # If there is an event loop running (the call
        # above doesn't raise an exception), we can
        # use nest_asyncio to patch the event loop.
        nest_asyncio.apply()

        return asyncio.run(coroutine)
    except RuntimeError:
        # Explicitly pass, since we want to fallback to asyncio.run
        pass

    # If no event loop is running, asyncio will
    # return a RuntimeError (https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.get_running_loop).
    # In that case, we can just use asyncio.run.
    return asyncio.run(coroutine)
