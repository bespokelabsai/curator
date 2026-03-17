"""In-memory request processor for low-latency small batch processing.

Bypasses the disk-based pipeline (JSONL write/read, Arrow files) and processes
requests entirely in memory. Designed for small batches where the overhead of
the full orchestration pipeline dominates actual API call time.

Activated only when CURATOR_DISABLE_CACHE=true is set.
"""

import asyncio
import datetime
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from datasets import Dataset
from pydantic import BaseModel

from bespokelabs.curator.log import logger
from bespokelabs.curator.request_processor.online.base_online_request_processor import APIRequest, BaseOnlineRequestProcessor
from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import _TokenUsage

if TYPE_CHECKING:
    from bespokelabs.curator.llm.prompt_formatter import PromptFormatter


@dataclass
class _InMemoryRequestResult:
    responses: List[GenericResponse]
    tracker: "_SharedStatusTracker"


@dataclass
class _DatasetBuildResult:
    dataset: Dataset
    failed_request_indices: list[int]


async def _process_requests_in_memory(
    processor: BaseOnlineRequestProcessor,
    requests: List[GenericRequest],
    prompt_formatter: "PromptFormatter",
) -> _InMemoryRequestResult:
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
        Responses and aggregate tracker stats for the batch
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
        total_requests=len(requests),
        max_requests_per_minute=rpm,
        max_tokens_per_minute=tpm,
        token_limit_strategy=processor.token_limit_strategy,
        compatible_provider=processor.compatible_provider,
        viewer_client=processor.viewer_client,
        model=prompt_formatter.model_name,
    )
    shared_tracker.num_tasks_started = len(requests)

    async def _call_single(
        session,
        generic_request: GenericRequest,
        idx: int,
    ) -> Optional[GenericResponse]:
        """Make a single API call with retries, respecting concurrency, RPM, and TPM limits."""
        async with semaphore:
            shared_tracker.num_tasks_in_progress += 1
            shared_tracker.max_concurrent_requests_seen = max(
                shared_tracker.max_concurrent_requests_seen,
                shared_tracker.num_tasks_in_progress,
            )

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
            try:
                for attempt in range(processor.config.max_retries + 1):
                    # Wait for RPM + TPM capacity (mirrors the normal path)
                    await shared_tracker.reserve_capacity(token_estimate)

                    try:
                        generic_response = await processor.call_single_request(
                            request=request,
                            session=session,
                            status_tracker=shared_tracker,
                        )

                        # Validate response format
                        if generic_response.finish_reason in processor.config.invalid_finish_reasons:
                            shared_tracker.update_stats(generic_response.token_usage, generic_response.response_cost)
                            raise ValueError(f"finish_reason was {generic_response.finish_reason}")

                        prompt_formatter.response_to_response_format(generic_response.response_message)

                        # Success mirrors the file-based path: only free over-estimation
                        # after the response passes validation.
                        shared_tracker.update_stats(generic_response.token_usage, generic_response.response_cost)
                        if generic_response.token_usage is not None:
                            shared_tracker.free_capacity(generic_response.token_usage, token_estimate)

                        shared_tracker.num_tasks_succeeded += 1
                        return generic_response

                    except Exception as e:
                        last_error = e
                        if attempt < processor.config.max_retries:
                            logger.warning(f"Request {idx} attempt {attempt + 1}/{processor.config.max_retries + 1} failed: {e}")
                            await asyncio.sleep(min(2**attempt, 10))

                # All retries exhausted
                logger.error(f"Request {idx} failed permanently: {last_error}")
                shared_tracker.num_tasks_failed += 1
                return GenericResponse(
                    response_message=None,
                    response_errors=[str(last_error)],
                    raw_request=api_specific_request,
                    raw_response=None,
                    generic_request=generic_request,
                    created_at=datetime.datetime.now(),
                    finished_at=datetime.datetime.now(),
                )
            finally:
                shared_tracker.num_tasks_in_progress -= 1

    tcp_limit = max_concurrent if rpm is None else rpm
    if processor.viewer_client is not None:
        await processor.viewer_client.session_inprogress()

    async with processor.aiohttp_connector(tcp_limit) as session:
        tasks = [_call_single(session, req, idx) for idx, req in enumerate(requests)]
        responses = await asyncio.gather(*tasks)

    if processor.viewer_client is not None:
        await processor.viewer_client.session_completed()

    return _InMemoryRequestResult(
        responses=[r for r in responses if r is not None],
        tracker=shared_tracker,
    )


def process_responses_to_dataset(
    responses: List[GenericResponse],
    prompt_formatter: "PromptFormatter",
    require_all_responses: bool = True,
    total_requests: int = 0,
) -> _DatasetBuildResult:
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
        Dataset and failed request indices

    Raises:
        ValueError: If require_all_responses is True and any requests failed,
                    if all requests failed, or if parse_func returns invalid rows.
    """
    from bespokelabs.curator.constants import _INTERNAL_PROMPT_KEY

    error_help = "Please check your `parse_func` is returning a valid row (dict) " "or list of rows (list of dicts) and re-run."

    rows = []
    failed_count = 0
    failed_requests = []
    failed_request_indices = []

    for response in responses:
        if response.response_errors is not None:
            failed_count += 1
            failed_requests.append(response)
            failed_request_indices.append(response.generic_request.original_row_idx)
            continue

        try:
            response_message = prompt_formatter.response_to_response_format(response.response_message)
        except Exception:
            logger.warning("Skipping response due to error parsing response into response format")
            failed_count += 1
            failed_requests.append(response)
            failed_request_indices.append(response.generic_request.original_row_idx)
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
                failed_request_indices.append(response.generic_request.original_row_idx)
                continue

            if not isinstance(parsed, list):
                parsed = [parsed]

            # Validate rows — invalid types / empty dicts are hard errors,
            # matching BaseRequestProcessor.create_dataset_files behaviour.
            produced_row = False
            for item in parsed:
                if isinstance(item, BaseModel):
                    item = item.model_dump()
                if not isinstance(item, dict):
                    raise ValueError(f"Got invalid row {item} of type {type(item)} from `parse_func`. " f"This should be type <class 'dict'>. {error_help}")
                if not item:
                    raise ValueError(f"Got empty row {item} from `parse_func`. {error_help}")
                rows.append(item)
                produced_row = True
            if not produced_row:
                failed_request_indices.append(response.generic_request.original_row_idx)
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

    return _DatasetBuildResult(
        dataset=Dataset.from_list(rows),
        failed_request_indices=failed_request_indices,
    )


class _SharedStatusTracker(OnlineStatusTracker):
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
        total_requests: int = 0,
        max_requests_per_minute: int | None = None,
        max_tokens_per_minute=None,
        token_limit_strategy=None,
        compatible_provider: str | None = None,
        viewer_client=None,
        model: str = "",
    ):
        super().__init__(
            total_requests=total_requests,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_limit_strategy=token_limit_strategy,
            compatible_provider=compatible_provider,
            viewer_client=viewer_client,
            model=model,
        )

        self._seconds_to_pause = seconds_to_pause
        self._capacity_lock = asyncio.Lock()

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
            self.last_update_time = time.time()

    # ------------------------------------------------------------------
    # RPM + TPM capacity
    # ------------------------------------------------------------------

    def update_display(self):
        """Disable Rich/tqdm tracker updates for the in-memory fast path."""
        return None

    async def reserve_capacity(self, token_estimate: _TokenUsage) -> None:
        """Block until RPM + TPM capacity is available, then reserve it."""
        while True:
            await self.wait_for_cooldown()
            async with self._capacity_lock:
                if self.has_capacity(token_estimate):
                    self.consume_capacity(token_estimate)
                    return
            await asyncio.sleep(0.1)
