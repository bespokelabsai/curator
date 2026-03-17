"""In-memory request processor for low-latency small batch processing.

Bypasses the disk-based pipeline (JSONL write/read, Arrow files) and processes
requests entirely in memory. Designed for small batches where the overhead of
the full orchestration pipeline dominates actual API call time.

Activated only when CURATOR_DISABLE_CACHE=true is set.
"""

import asyncio
import datetime
import time
from typing import TYPE_CHECKING, List, Optional

import aiohttp
from datasets import Dataset
from pydantic import BaseModel

from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.online.base_online_request_processor import APIRequest, BaseOnlineRequestProcessor
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import _TokenUsage

if TYPE_CHECKING:
    from bespokelabs.curator.llm.prompt_formatter import PromptFormatter


async def _process_requests_in_memory(
    processor: BaseOnlineRequestProcessor,
    requests: List[GenericRequest],
    prompt_formatter: "PromptFormatter",
) -> List[GenericResponse]:
    """Process a list of GenericRequests in memory using the processor's API call logic.

    Respects the processor's concurrency limits (max_concurrent_requests),
    RPM throttling, and TPM token-capacity gating. A shared status tracker
    propagates 429 cooldowns across all in-flight requests so that a
    rate-limit error on one request pauses the entire batch — matching the
    behaviour of the standard file-based pipeline.

    Args:
        processor: The online request processor (OpenAI, Anthropic, LiteLLM, etc.)
        requests: List of GenericRequest objects to process
        prompt_formatter: Formatter for prompts and responses

    Returns:
        List of GenericResponse objects
    """
    # Determine concurrency limit from processor config
    max_concurrent = processor.max_concurrent_requests
    if max_concurrent is None:
        max_concurrent = min(len(requests), 100)
    semaphore = asyncio.Semaphore(int(max_concurrent))

    # Shared tracker so a 429 on any request triggers a global cooldown,
    # and token capacity is tracked across all concurrent requests.
    rpm = processor.max_requests_per_minute
    tpm = processor.max_tokens_per_minute
    shared_tracker = _SharedStatusTracker(
        seconds_to_pause=processor.config.seconds_to_pause_on_rate_limit,
        max_requests_per_minute=rpm,
        max_tokens_per_minute=tpm,
    )

    async def _call_single(
        session: aiohttp.ClientSession,
        generic_request: GenericRequest,
        idx: int,
    ) -> Optional[GenericResponse]:
        """Make a single API call with retries, respecting concurrency, RPM, and TPM limits."""
        async with semaphore:
            generic_request = processor._unpack_multimodal(generic_request)
            api_specific_request = processor.create_api_specific_request_online(generic_request)

            request = APIRequest(
                task_id=idx,
                generic_request=generic_request,
                api_specific_request=api_specific_request,
                attempts_left=processor.config.max_retries,
                prompt_formatter=prompt_formatter,
            )

            # Estimate tokens for capacity gating (mirrors the normal path)
            if tpm is not None:
                token_estimate = processor.estimate_total_tokens(generic_request.messages)
            else:
                token_estimate = _TokenUsage()

            last_error = None
            for attempt in range(processor.config.max_retries + 1):
                # Honour shared cooldown from 429s on other requests
                await shared_tracker.wait_for_cooldown()

                # Wait for RPM + TPM capacity (mirrors has_capacity loop)
                await shared_tracker.wait_for_capacity(token_estimate)
                shared_tracker.consume_capacity(token_estimate)

                try:
                    generic_response = await processor.call_single_request(
                        request=request,
                        session=session,
                        status_tracker=shared_tracker,
                    )

                    # Free the over-estimated capacity; actual usage is in the response
                    if generic_response.token_usage is not None:
                        shared_tracker.free_capacity(generic_response.token_usage, token_estimate)

                    # Validate response format
                    if generic_response.finish_reason in processor.config.invalid_finish_reasons:
                        raise ValueError(f"finish_reason was {generic_response.finish_reason}")

                    prompt_formatter.response_to_response_format(generic_response.response_message)
                    return generic_response

                except Exception as e:
                    last_error = e
                    # Free blocked capacity on failure
                    shared_tracker.free_capacity(_TokenUsage(), token_estimate)
                    if attempt < processor.config.max_retries:
                        logger.warning(f"Request {idx} attempt {attempt + 1}/{processor.config.max_retries + 1} failed: {e}")
                        await asyncio.sleep(min(2**attempt, 10))

            # All retries exhausted
            logger.error(f"Request {idx} failed permanently: {last_error}")
            return GenericResponse(
                response_message=None,
                response_errors=[str(last_error)],
                raw_request=api_specific_request,
                raw_response=None,
                generic_request=generic_request,
                created_at=datetime.datetime.now(),
                finished_at=datetime.datetime.now(),
            )

    connector = aiohttp.TCPConnector(limit=min(len(requests) * 2, 100))
    timeout = aiohttp.ClientTimeout(total=processor.config.request_timeout)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [_call_single(session, req, idx) for idx, req in enumerate(requests)]
        responses = await asyncio.gather(*tasks)

    return [r for r in responses if r is not None]


def process_responses_to_dataset(
    responses: List[GenericResponse],
    prompt_formatter: "PromptFormatter",
    require_all_responses: bool = True,
    total_requests: int = 0,
) -> Dataset:
    """Build a HuggingFace Dataset directly from in-memory responses.

    Mirrors the validation behaviour of
    ``BaseRequestProcessor.create_dataset_files``: non-dict rows and empty
    dicts from ``parse_func`` raise ``ValueError`` immediately (they are
    *not* caught and silently counted as failures).

    Args:
        responses: List of GenericResponse objects
        prompt_formatter: Formatter for parsing responses
        require_all_responses: If True, raise when any request failed
        total_requests: Total number of requests submitted

    Returns:
        Dataset containing processed responses

    Raises:
        ValueError: If require_all_responses is True and any requests failed,
                    if all requests failed, or if parse_func returns invalid rows.
    """
    from bespokelabs.curator.constants import _INTERNAL_PROMPT_KEY

    error_help = "Please check your `parse_func` is returning a valid row (dict) " "or list of rows (list of dicts) and re-run."

    rows = []
    failed_count = 0
    failed_requests = []

    for response in responses:
        if response.response_errors is not None:
            failed_count += 1
            failed_requests.append(response)
            continue

        try:
            response_message = prompt_formatter.response_to_response_format(response.response_message)
        except Exception:
            logger.warning("Skipping response due to error parsing response into response format")
            failed_count += 1
            failed_requests.append(response)
            continue

        if prompt_formatter.parse_func:
            # Call parse_func — exceptions here are user errors, counted as failures
            try:
                if _INTERNAL_PROMPT_KEY in response.generic_request.original_row:
                    row = response.generic_request.original_row[_INTERNAL_PROMPT_KEY]
                else:
                    row = response.generic_request.original_row
                parsed = prompt_formatter.parse_func(row, response_message)
            except Exception as e:
                logger.warning(f"Skipping response due to error in `parse_func` :: {e}")
                failed_count += 1
                failed_requests.append(response)
                continue

            if not isinstance(parsed, list):
                parsed = [parsed]

            # Validate rows — invalid types / empty dicts are hard errors,
            # matching BaseRequestProcessor.create_dataset_files behaviour.
            for item in parsed:
                if isinstance(item, BaseModel):
                    item = item.model_dump()
                if not isinstance(item, dict):
                    raise ValueError(f"Got invalid row {item} of type {type(item)} from `parse_func`. " f"This should be type <class 'dict'>. {error_help}")
                if not item:
                    raise ValueError(f"Got empty row {item} from `parse_func`. {error_help}")
                rows.append(item)
        else:
            response_value = response_message
            if hasattr(response_value, "model_dump"):
                response_value = response_value.model_dump()
            elif hasattr(response_value, "__dict__"):
                response_value = response_value.__dict__
            rows.append({"response": response_value})

    if not rows:
        error_sample = [str(r.response_errors) for r in failed_requests[:10]]
        raise ValueError(f"All {failed_count} requests failed. No successful responses to create dataset from.\n" f"Sample errors: {error_sample}")

    if failed_count > 0:
        logger.warning(f"{failed_count}/{total_requests or len(responses)} requests failed in fast path processing.")
        if require_all_responses:
            error_sample = [str(r.response_errors) for r in failed_requests[:10]]
            raise ValueError(f"{failed_count} requests failed and require_all_responses is True.\n" f"Sample errors: {error_sample}")

    return Dataset.from_list(rows)


class _SharedStatusTracker:
    """Shared status tracker across all in-flight requests in a fast-path batch.

    Provides the same attributes that ``call_single_request`` writes to on
    the various provider processors (``num_api_errors``,
    ``time_of_last_rate_limit_error``, etc.), plus lightweight RPM and TPM
    capacity gating that mirrors ``OnlineStatusTracker.has_capacity`` /
    ``consume_capacity`` / ``free_capacity`` without the Rich/tqdm overhead.
    """

    def __init__(
        self,
        seconds_to_pause: int = 10,
        max_requests_per_minute: int | None = None,
        max_tokens_per_minute=None,
    ):
        # Fields written by call_single_request in provider processors
        self.num_api_errors = 0
        self.num_rate_limit_errors = 0
        self.num_other_errors = 0
        self.time_of_last_rate_limit_error: float = 0
        self.max_concurrent_requests_seen = 0

        self._seconds_to_pause = seconds_to_pause

        # RPM capacity
        self._max_rpm = max_requests_per_minute
        self._available_request_capacity: float = float(max_requests_per_minute) if max_requests_per_minute else 0
        self._last_rpm_update: float = time.time()

        # TPM capacity — may be an int (combined) or _TokenUsage (separate)
        self._max_tpm = max_tokens_per_minute
        if isinstance(max_tokens_per_minute, _TokenUsage):
            self._available_token_capacity = _TokenUsage(
                input=max_tokens_per_minute.input,
                output=max_tokens_per_minute.output,
            )
        elif max_tokens_per_minute is not None:
            self._available_token_capacity: int | _TokenUsage = int(max_tokens_per_minute)
        else:
            self._available_token_capacity = None
        self._last_tpm_update: float = time.time()

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------

    async def wait_for_cooldown(self) -> None:
        """Sleep if a rate-limit error was recently recorded."""
        if self.time_of_last_rate_limit_error == 0:
            return
        elapsed = time.time() - self.time_of_last_rate_limit_error
        remaining = self._seconds_to_pause - elapsed
        if remaining > 0:
            logger.warning(f"Fast-path rate-limit cooldown: pausing {int(remaining)}s")
            await asyncio.sleep(remaining)

    # ------------------------------------------------------------------
    # RPM + TPM capacity
    # ------------------------------------------------------------------

    def _refresh_capacity(self) -> None:
        """Replenish capacity based on elapsed time (token-bucket style)."""
        now = time.time()

        if self._max_rpm:
            elapsed = now - self._last_rpm_update
            self._available_request_capacity = min(
                self._available_request_capacity + self._max_rpm * elapsed / 60.0,
                float(self._max_rpm),
            )
            self._last_rpm_update = now

        if self._max_tpm is not None and self._available_token_capacity is not None:
            elapsed = now - self._last_tpm_update
            if isinstance(self._max_tpm, _TokenUsage):
                self._available_token_capacity.input = min(
                    self._available_token_capacity.input + int(self._max_tpm.input * elapsed / 60.0),
                    self._max_tpm.input,
                )
                self._available_token_capacity.output = min(
                    self._available_token_capacity.output + int(self._max_tpm.output * elapsed / 60.0),
                    self._max_tpm.output,
                )
            else:
                total = token_total(self._available_token_capacity)
                max_total = token_total(self._max_tpm)
                self._available_token_capacity = min(
                    total + int(max_total * elapsed / 60.0),
                    max_total,
                )
            self._last_tpm_update = now

    def _has_capacity(self, token_estimate: _TokenUsage) -> bool:
        """Check if there is RPM + TPM capacity for one more request."""
        self._refresh_capacity()

        # RPM check
        if self._max_rpm and self._available_request_capacity < 1:
            return False

        # TPM check
        if self._available_token_capacity is not None and self._max_tpm is not None:
            if isinstance(self._available_token_capacity, _TokenUsage):
                if (token_estimate.input or 0) > (self._available_token_capacity.input or 0):
                    return False
                if (token_estimate.output or 0) > (self._available_token_capacity.output or 0):
                    return False
            else:
                needed = (token_estimate.input or 0) + (token_estimate.output or 0)
                if needed > token_total(self._available_token_capacity):
                    return False

        return True

    async def wait_for_capacity(self, token_estimate: _TokenUsage) -> None:
        """Block until RPM + TPM capacity is available."""
        while not self._has_capacity(token_estimate):
            await asyncio.sleep(0.1)

    def consume_capacity(self, token_estimate: _TokenUsage) -> None:
        """Reserve capacity before dispatching a request."""
        if self._max_rpm:
            self._available_request_capacity -= 1

        if self._available_token_capacity is not None:
            if isinstance(self._available_token_capacity, _TokenUsage):
                self._available_token_capacity.input -= token_estimate.input or 0
                self._available_token_capacity.output -= token_estimate.output or 0
            else:
                self._available_token_capacity -= (token_estimate.input or 0) + (token_estimate.output or 0)

    def free_capacity(self, used: _TokenUsage, blocked: _TokenUsage) -> None:
        """Return over-estimated capacity after a request completes."""
        if self._available_token_capacity is None:
            return
        if isinstance(self._available_token_capacity, _TokenUsage):
            freed_in = (blocked.input or 0) - (used.input or 0)
            freed_out = (blocked.output or 0) - (used.output or 0)
            if freed_in > 0:
                self._available_token_capacity.input += freed_in
            if freed_out > 0:
                self._available_token_capacity.output += freed_out
        else:
            freed = ((blocked.input or 0) + (blocked.output or 0)) - ((used.input or 0) + (used.output or 0))
            if freed > 0:
                self._available_token_capacity += freed


def token_total(value) -> int:
    """Extract a scalar total from an int or _TokenUsage."""
    if isinstance(value, _TokenUsage):
        return (value.input or 0) + (value.output or 0)
    return int(value)
