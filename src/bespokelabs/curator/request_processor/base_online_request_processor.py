from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import datetime
import time
from typing import Optional
from tqdm import tqdm
import logging
import asyncio
import aiohttp
import os
import json
import resource

from bespokelabs.curator.dataset import Dataset
from bespokelabs.curator.request_processor.base_request_processor import BaseRequestProcessor
from bespokelabs.curator.prompter.prompter import PromptFormatter
from bespokelabs.curator.request_processor.generic_request import GenericRequest
from bespokelabs.curator.request_processor.event_loop import run_in_event_loop
from bespokelabs.curator.request_processor.generic_response import GenericResponse, TokenUsage
import aiofiles
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    Group,
)
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from litellm import model_cost

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_REQUESTS_PER_MINUTE = 10
DEFAULT_TOKENS_PER_MINUTE = 1_000


@dataclass
class StatusTracker:
    """Tracks the status of all requests."""
    model: str = field(default=None, init=False)
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_tasks_already_completed: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    num_rate_limit_errors: int = 0
    available_request_capacity: float = 0
    available_token_capacity: float = 0
    last_update_time: float = field(default_factory=time.time)
    max_requests_per_minute: int = 0
    max_tokens_per_minute: int = 0
    time_of_last_rate_limit_error: float = field(default=None)
    
    # Rich progress display
    progress: Progress = field(default=None, init=False)
    task_id: int = field(default=None, init=False)
    console: Console = field(default_factory=Console, init=False)
    
    # Stats tracking
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0
    
    # Add new fields for cost per million tokens
    input_cost_per_million: float = 0
    output_cost_per_million: float = 0

    def __post_init__(self):
        """Initialize the rich progress display."""
        # Create the progress display
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=50),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold white]•[/bold white]"),
            TimeElapsedColumn(),
            TextColumn("[bold white]•[/bold white]"),
            TimeRemainingColumn(),
            TextColumn(
                "{task.fields[status_text]}\n"
                "{task.fields[token_text]}\n"
                "{task.fields[cost_text]}\n"
                "{task.fields[model_name_text]}\n"
                "{task.fields[rate_limit_text]}\n"
                "{task.fields[price_text]}",
                justify="left",
            ),
            expand=True,
            refresh_per_second=1,
        )
        
        self.task_id = None
        self.progress.start()

    @property
    def total(self) -> int:
        """Get total tasks from progress."""
        if self.progress and self.task_id is not None:
            return self.progress.tasks[self.task_id].total
        return 0
        
    def initialize_progress(self, total_requests: int, processor_name: str):
        """Initialize the progress bar with total requests."""
        # Initialize cost rates here, after model has been set
        if self.model in model_cost:
            self.input_cost_per_million = model_cost[self.model]['input_cost_per_token'] * 1_000_000
            self.output_cost_per_million = model_cost[self.model]['output_cost_per_token'] * 1_000_000
        
        self.task_id = self.progress.add_task(
            description=f"[cyan]{processor_name}",
            total=total_requests,
            completed=0,
            status_text="[bold white]Status:[/bold white] [dim]Initializing...[/dim]",
            token_text="[bold white]Tokens:[/bold white] --",
            cost_text="[bold white]Cost:[/bold white] --",
            model_name_text="[bold white]Model:[/bold white] --",
            rate_limit_text="[bold white]Rate Limits:[/bold white] --",
            price_text="[bold white]Model Pricing:[/bold white] --"
        )

    def update_stats(self, token_usage: TokenUsage, cost: float):
        """Update token and cost statistics."""
        if token_usage:
            self.total_prompt_tokens += token_usage.prompt_tokens
            self.total_completion_tokens += token_usage.completion_tokens
            self.total_tokens += token_usage.total_tokens
        if cost:
            self.total_cost += cost
        self.update_progress_display()

    def update_progress_display(self):
        """Update the rich progress display with current statistics."""
        if not self.progress or self.task_id is None:
            return
        # Calculate averages and stats
        avg_prompt = self.total_prompt_tokens / max(1, self.num_tasks_succeeded)
        avg_completion = self.total_completion_tokens / max(1, self.num_tasks_succeeded)
        avg_total = self.total_tokens / max(1, self.num_tasks_succeeded)
        avg_cost = self.total_cost / max(1, self.num_tasks_succeeded)
        projected_cost = avg_cost * self.total
        
        # Calculate cost per hour
        elapsed_hours = max(0.001, self.progress.tasks[self.task_id].elapsed) / 3600
        cost_per_hour = self.total_cost / elapsed_hours if elapsed_hours > 0 else 0

        # Format the text for each line
        status_text = (
            "[bold white]Status:[/bold white] Processing "
            f"[dim]([green]✓{self.num_tasks_succeeded}[/green] "
            f"[red]✗{self.num_tasks_failed}[/red] "
            f"[yellow]⋯{self.num_tasks_in_progress}[/yellow])[/dim]"
        )

        token_text = (
            "[bold white]Tokens:[/bold white] "
            f"Avg Input: [blue]{avg_prompt:.0f}[/blue] • "
            f"Avg Output: [blue]{avg_completion:.0f}[/blue] • "
            f"Total: [blue]{avg_total:.0f}[/blue] • "
            f"Cumulative: [blue]{self.total_tokens:,}[/blue]"
        )

        cost_text = (
            "[bold white]Cost:[/bold white] "
            f"Current: [magenta]${self.total_cost:.3f}[/magenta] • "
            f"Projected: [magenta]${projected_cost:.3f}[/magenta] • "
            f"Rate: [magenta]${cost_per_hour:.3f}/hr[/magenta] • "
        )
        model_name_text = (
            f"[bold white]Model:[/bold white] [blue]{self.model}[/blue]"
        )
        rate_limit_text = (
            f"[bold white]Rate Limits:[/bold white] "
            f"rpm: [blue]{self.max_requests_per_minute}[/blue] • "
            f"tpm: [blue]{self.max_tokens_per_minute}[/blue]"
        )
        price_text = (
            "[bold white]Model Pricing:[/bold white] "
            f"Per 1M tokens: Input: [red]${self.input_cost_per_million:.3f}[/red] • Output: [red]${self.output_cost_per_million:.3f}[/red]"
        )

        self.progress.update(
            self.task_id,
            advance=1,
            completed=self.num_tasks_succeeded,
            status_text=status_text,
            token_text=token_text,
            cost_text=cost_text,
            model_name_text=model_name_text,
            rate_limit_text=rate_limit_text,
            price_text=price_text,
        )

    def __del__(self):
        """Ensure progress bar is stopped on deletion."""
        if self.progress:
            self.progress.stop()

    def stop(self):
        """Stop the progress display and show final statistics."""
        if self.progress:
            self.progress.stop()
            
            # Create final statistics table
            table = Table(title="Final Curator Statistics", box=box.ROUNDED)
            table.add_column("Section/Metric", style="cyan")
            table.add_column("Value", style="yellow")
            
            # Request Statistics
            table.add_row("Requests", "", style="bold magenta")
            table.add_row("Total Processed", str(self.num_tasks_succeeded + self.num_tasks_failed))
            table.add_row("Successful", f"[green]{self.num_tasks_succeeded}[/green]")
            table.add_row("Failed", f"[red]{self.num_tasks_failed}[/red]")
            
            # Token Statistics
            table.add_row("Tokens", "", style="bold magenta") 
            table.add_row("Total Tokens Used", f"{self.total_tokens:,}")
            table.add_row("Total Prompt Tokens", f"{self.total_prompt_tokens:,}")
            table.add_row("Total Completion Tokens", f"{self.total_completion_tokens:,}")
            table.add_row("Average Tokens per Request", f"{self.total_tokens / max(1, self.num_tasks_succeeded):.1f}")
            table.add_row("Average Prompt Tokens", f"{self.total_prompt_tokens / max(1, self.num_tasks_succeeded):.1f}")
            table.add_row("Average Completion Tokens", f"{self.total_completion_tokens / max(1, self.num_tasks_succeeded):.1f}")
            
            # Cost Statistics
            table.add_row("Costs", "", style="bold magenta")
            table.add_row("Total Cost", f"${self.total_cost:.4f}")
            table.add_row("Average Cost per Request", f"${self.total_cost / max(1, self.num_tasks_succeeded):.4f}")
            table.add_row("Input Cost per 1M Tokens", f"${self.input_cost_per_million:.4f}")
            table.add_row("Output Cost per 1M Tokens", f"${self.output_cost_per_million:.4f}")            
            self.console.print(table)

    def __str__(self):
        return (
            f"Tasks - Started: {self.num_tasks_started}, "
            f"In Progress: {self.num_tasks_in_progress}, "
            f"Succeeded: {self.num_tasks_succeeded}, "
            f"Failed: {self.num_tasks_failed}, "
            f"Already Completed: {self.num_tasks_already_completed}\n"
            f"Errors - API: {self.num_api_errors}, "
            f"Rate Limit: {self.num_rate_limit_errors}, "
            f"Other: {self.num_other_errors}"
        )

    def update_capacity(self):
        """Update available capacity based on time elapsed"""
        current_time = time.time()
        seconds_since_update = current_time - self.last_update_time

        self.available_request_capacity = min(
            self.available_request_capacity
            + self.max_requests_per_minute * seconds_since_update / 60.0,
            self.max_requests_per_minute,
        )

        self.available_token_capacity = min(
            self.available_token_capacity
            + self.max_tokens_per_minute * seconds_since_update / 60.0,
            self.max_tokens_per_minute,
        )

        self.last_update_time = current_time

    def has_capacity(self, token_estimate: int) -> bool:
        """Check if there's enough capacity for a request"""
        self.update_capacity()
        has_capacity = (
            self.available_request_capacity >= 1 and self.available_token_capacity >= token_estimate
        )
        if not has_capacity:
            logger.debug(
                f"No capacity for request with {token_estimate} tokens. "
                f"Available capacity: {self.available_token_capacity} tokens, "
                f"{self.available_request_capacity} requests."
            )
        return has_capacity

    def consume_capacity(self, token_estimate: int):
        """Consume capacity for a request"""
        self.available_request_capacity -= 1
        self.available_token_capacity -= token_estimate

    def update_error_state(self, error_message: str):
        """Update the display to show an error state."""
        if not self.progress or self.task_id is None:
            return
        
        self.progress.update(
            self.task_id,
            status_text=f"[bold red]Error:[/bold red] {error_message}",
            token_text="[dim]Process failed[/dim]",
            cost_text="[dim]Process failed[/dim]",
            model_name_text="[dim]Process failed[/dim]",
            rate_limit_text="[dim]Process failed[/dim]",
            price_text="[dim]Process failed[/dim]"
        )


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata."""

    task_id: int
    generic_request: GenericRequest
    api_specific_request: dict
    attempts_left: int
    result: list = field(default_factory=list)
    prompt_formatter: PromptFormatter = field(default=None)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)


class BaseOnlineRequestProcessor(BaseRequestProcessor, ABC):
    """Abstract base class for online request processors that make real-time API calls."""

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_requests_per_minute: Optional[int] = None,
        max_tokens_per_minute: Optional[int] = None,
    ):
        super().__init__(batch_size=None)
        self.model: str = model
        self.temperature: float | None = temperature
        self.top_p: float | None = top_p
        self.presence_penalty: float | None = presence_penalty
        self.frequency_penalty: float | None = frequency_penalty
        self.prompt_formatter: Optional[PromptFormatter] = None
        self.max_requests_per_minute: Optional[int] = max_requests_per_minute
        self.max_tokens_per_minute: Optional[int] = max_tokens_per_minute
        self.DEFAULT_MAX_REQUESTS_PER_MINUTE = DEFAULT_REQUESTS_PER_MINUTE
        self.DEFAULT_MAX_TOKENS_PER_MINUTE = DEFAULT_TOKENS_PER_MINUTE

    def get_rate_limit(self, name, header_value):
        """Uses manual values if set, otherwise uses headers if available, and if not available uses defaults."""
        manual_value = getattr(self, name)
        default_value = getattr(self, f"DEFAULT_{name.upper()}")
        if manual_value is not None:
            logger.info(f"Manually set {name} to {manual_value}")
            return manual_value
        elif header_value != 0:
            logger.info(f"Automatically set {name} to {header_value}")
            return header_value
        else:
            logger.warning(
                f"No manual {name} set, and headers based detection failed, using default value of {default_value}"
            )
            return default_value

    def get_rate_limits(self) -> dict:
        """Get rate limits for the API. Returns a dictionary with max_requests_per_minute and max_tokens_per_minute"""

        # Get values from headers
        header_based_rate_limits = self.get_header_based_rate_limits()
        header_tpm = header_based_rate_limits["max_tokens_per_minute"]
        header_rpm = header_based_rate_limits["max_requests_per_minute"]

        # Determine final rate limit
        tpm = self.get_rate_limit("max_tokens_per_minute", header_tpm)
        rpm = self.get_rate_limit("max_requests_per_minute", header_rpm)

        return {"max_requests_per_minute": rpm, "max_tokens_per_minute": tpm}

    @abstractmethod
    def get_header_based_rate_limits(self) -> dict:
        """Get rate limits for the API from headers. Returns a dictionary with max_requests_per_minute and max_tokens_per_minute"""
        pass

    @abstractmethod
    def estimate_total_tokens(self, messages: list) -> int:
        """Estimate total tokens for a request"""
        pass

    @abstractmethod
    def estimate_output_tokens(self) -> int:
        """Estimate output tokens for a request"""
        pass

    def check_structured_output_support(self) -> bool:
        """Check if the model supports structured output"""
        return True

    def run(
        self,
        dataset: Optional[Dataset],
        working_dir: str,
        parse_func_hash: str,
        prompt_formatter: PromptFormatter,
    ) -> Dataset:
        """Run completions using the online API with async processing."""
        logger.info(f"Running {self.__class__.__name__} completions with model: {self.model}")

        self.prompt_formatter = prompt_formatter
        if self.prompt_formatter.response_format:
            if not self.check_structured_output_support():
                raise ValueError(
                    f"Model {self.model} does not support structured output, "
                    f"response_format: {self.prompt_formatter.response_format}"
                )
        generic_request_files = self.create_request_files(dataset, working_dir, prompt_formatter)
        generic_responses_files = [
            f"{working_dir}/responses_{i}.jsonl" for i in range(len(generic_request_files))
        ]

        for request_file, response_file in zip(generic_request_files, generic_responses_files):
            run_in_event_loop(
                self.process_requests_from_file(
                    generic_request_filepath=request_file,
                    save_filepath=response_file,
                    max_attempts=5,
                    resume=True,
                )
            )

        return self.create_dataset_files(working_dir, parse_func_hash, prompt_formatter)

    async def process_requests_from_file(
        self,
        generic_request_filepath: str,
        save_filepath: str,
        max_attempts: int,
        resume: bool,
        resume_no_retry: bool = False,
    ) -> None:
        """Processes API requests in parallel, throttling to stay under rate limits."""
        # Initialize trackers
        status_tracker = StatusTracker()
        status_tracker.model = self.model
        try:
            # Count total requests
            total_requests = sum(1 for _ in open(generic_request_filepath))
            
            # Initialize progress display
            status_tracker.initialize_progress(total_requests, self.__class__.__name__)

            try:
                # Get rate limits
                rate_limits = self.get_rate_limits()
                status_tracker.max_requests_per_minute = rate_limits["max_requests_per_minute"]
                status_tracker.max_tokens_per_minute = rate_limits["max_tokens_per_minute"]
            except Exception as e:
                # Update status to show error
                status_tracker.update_error_state(
                    f"Failed to initialize: {str(e).split('litellm.exceptions.')[-1]}"
                )
                status_tracker.stop()
                raise

            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(
                resource.RLIMIT_NOFILE,
                (min(hard, int(10 * status_tracker.max_requests_per_minute)), hard),
            )

            # Track completed requests for resume functionality
            completed_request_ids = set()
            if os.path.exists(save_filepath):
                if resume:
                    logger.debug(f"Resuming progress from existing file: {save_filepath}")
                    logger.debug(
                        f"Removing all failed requests from {save_filepath} so they can be retried"
                    )
                    temp_filepath = f"{save_filepath}.temp"
                    num_previously_failed_requests = 0

                    with open(save_filepath, "r") as input_file, open(
                        temp_filepath, "w"
                    ) as output_file:
                        for line in input_file:
                            response = GenericResponse.model_validate_json(line)
                            if response.response_errors:
                                logger.debug(
                                    f"Request {response.generic_request.original_row_idx} previously failed due to errors: "
                                    f"{response.response_errors}, removing from output and will retry"
                                )
                                num_previously_failed_requests += 1
                            else:
                                completed_request_ids.add(response.generic_request.original_row_idx)
                                output_file.write(line)

                    logger.info(
                        f"Found {len(completed_request_ids)} completed requests and "
                        f"{num_previously_failed_requests} previously failed requests"
                    )
                    logger.info("Failed requests and remaining requests will now be processed.")
                    os.replace(temp_filepath, save_filepath)

                elif resume_no_retry:
                    logger.warning(
                        f"Resuming progress from existing file: {save_filepath}, without retrying failed requests"
                    )
                    num_previously_failed_requests = 0

                    with open(save_filepath, "r") as input_file:
                        for line in input_file:
                            response = GenericResponse.model_validate_json(line)
                            if response.response_errors:
                                logger.debug(
                                    f"Request {response.generic_request.original_row_idx} previously failed due to errors: "
                                    f"{response.response_errors}, will NOT retry"
                                )
                                num_previously_failed_requests += 1
                            completed_request_ids.add(response.generic_request.original_row_idx)

                    logger.info(
                        f"Found {len(completed_request_ids)} total requests and "
                        f"{num_previously_failed_requests} previously failed requests"
                    )
                    logger.info("Remaining requests will now be processed.")

                else:
                    user_input = input(
                        f"File {save_filepath} already exists.\n"
                        f"To resume if there are remaining requests without responses, run with --resume flag.\n"
                        f"Overwrite? (Y/n): "
                    )
                    if user_input.lower() not in ["y", ""]:
                        logger.info("Aborting operation.")
                        return

            # Initialize retry queue
            queue_of_requests_to_retry = asyncio.Queue()

            # Use higher connector limit for better throughput
            connector = aiohttp.TCPConnector(limit=10 * status_tracker.max_requests_per_minute)
            async with aiohttp.ClientSession(
                connector=connector
            ) as session:  # Initialize ClientSession here
                async with aiofiles.open(generic_request_filepath) as file:
                    pending_requests = []

                    async for line in file:
                        generic_request = GenericRequest.model_validate_json(line)

                        if resume and generic_request.original_row_idx in completed_request_ids:
                            status_tracker.num_tasks_already_completed += 1
                            status_tracker.num_tasks_succeeded += 1
                            status_tracker.update_progress_display()
                            continue

                        request = APIRequest(
                            task_id=status_tracker.num_tasks_started,
                            generic_request=generic_request,
                            api_specific_request=self.create_api_specific_request(generic_request),
                            attempts_left=max_attempts,
                            prompt_formatter=self.prompt_formatter,
                        )

                        token_estimate = self.estimate_total_tokens(request.generic_request.messages)

                        # Wait for capacity if needed
                        while not status_tracker.has_capacity(token_estimate):
                            await asyncio.sleep(0.1)

                        # Consume capacity before making request
                        status_tracker.consume_capacity(token_estimate)

                        task = asyncio.create_task(
                            self.handle_single_request_with_retries(
                                request=request,
                                session=session,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        pending_requests.append(task)

                        status_tracker.num_tasks_started += 1
                        status_tracker.num_tasks_in_progress += 1

                # Wait for all tasks to complete
                if pending_requests:
                    await asyncio.gather(*pending_requests)

                # Process any remaining retries in the queue
                pending_retries = set()
                while not queue_of_requests_to_retry.empty() or pending_retries:
                    # Process new items from the queue if we have capacity
                    if not queue_of_requests_to_retry.empty():
                        retry_request = await queue_of_requests_to_retry.get()
                        token_estimate = self.estimate_total_tokens(
                            retry_request.generic_request.messages
                        )
                        attempt_number = 6 - retry_request.attempts_left
                        logger.info(
                            f"Processing retry for request {retry_request.task_id} "
                            f"(attempt #{attempt_number} of 5). "
                            f"Previous errors: {retry_request.result}"
                        )

                        # Wait for capacity if needed
                        while not status_tracker.has_capacity(token_estimate):
                            await asyncio.sleep(0.1)

                        # Consume capacity before making request
                        status_tracker.consume_capacity(token_estimate)

                        task = asyncio.create_task(
                            self.handle_single_request_with_retries(
                                request=retry_request,
                                session=session,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        pending_retries.add(task)

                    # Wait for some tasks to complete
                    if pending_retries:
                        done, pending_retries = await asyncio.wait(pending_retries, timeout=0.1)

            status_tracker.stop()

            # Log final status
            logger.info(f"Processing complete. Results saved to {save_filepath}")
            logger.info(f"Status tracker: {status_tracker}")

            if status_tracker.num_tasks_failed > 0:
                logger.warning(
                    f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} "
                    f"requests failed. Errors logged to {save_filepath}."
                )

        except Exception as e:
            logger.error(f"Error processing requests: {str(e)}")
            status_tracker.update_error_state(f"Error processing requests: {str(e)}")
            status_tracker.stop()
            raise

    async def handle_single_request_with_retries(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ) -> None:
        """Common wrapper for handling a single request with error handling and retries.

        This method implements the common try/except logic and retry mechanism,
        while delegating the actual API call to call_single_request.

        Args:
            request (APIRequest): The request to process
            session (aiohttp.ClientSession): Async HTTP session
            retry_queue (asyncio.Queue): Queue for failed requests
            save_filepath (str): Path to save responses
            status_tracker (StatusTracker): Tracks request status
        """
        try:
            generic_response = await self.call_single_request(
                request=request,
                session=session,
                status_tracker=status_tracker,
            )

            # Save response in the base class
            await self.append_generic_response(generic_response, save_filepath)

            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            status_tracker.update_progress_display()

        except Exception as e:
            logger.warning(
                f"Request {request.task_id} failed with Exception {e}, attempts left {request.attempts_left}"
            )
            status_tracker.num_other_errors += 1
            request.result.append(e)

            if request.attempts_left > 0:
                request.attempts_left -= 1
                logger.info(
                    f"Adding request {request.task_id} to retry queue. Will retry in next available slot. "
                    f"Attempts remaining: {request.attempts_left}"
                )
                retry_queue.put_nowait(request)
            else:
                logger.error(
                    f"Request {request.task_id} failed permanently after exhausting all 5 retry attempts. "
                    f"Errors: {[str(e) for e in request.result]}"
                )
                generic_response = GenericResponse(
                    response_message=None,
                    response_errors=[str(e) for e in request.result],
                    raw_request=request.api_specific_request,
                    raw_response=None,
                    generic_request=request.generic_request,
                    created_at=request.created_at,
                    finished_at=datetime.datetime.now(),
                )
                
                await self.append_generic_response(generic_response, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
            # Make sure to update progress display for failed requests too
            status_tracker.update_progress_display()


    @abstractmethod
    async def call_single_request(
        self,
        request: APIRequest,
        session: aiohttp.ClientSession,
        status_tracker: StatusTracker,
    ) -> GenericResponse:
        """Make a single API request without error handling.

        This method should implement the actual API call logic
        without handling retries or errors.

        Args:
            request (APIRequest): Request to process
            session (aiohttp.ClientSession): Async HTTP session
            status_tracker (StatusTracker): Tracks request status

        Returns:
            GenericResponse: The response from the API call
        """
        pass

    async def append_generic_response(self, data: GenericResponse, filename: str) -> None:
        """Append a response to a jsonl file with async file operations."""
        json_string = json.dumps(data.model_dump(), default=str)
        async with aiofiles.open(filename, "a") as f:
            await f.write(json_string + "\n")
        logger.debug(f"Successfully appended response to {filename}")
