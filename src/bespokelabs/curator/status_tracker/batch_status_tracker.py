import logging
from dataclasses import dataclass, field

from bespokelabs.curator.request_processor.generic_batch_object import GenericBatchObject

logger = logging.getLogger(__name__)


@dataclass
class BatchStatusTracker:
    # total number of requests in all request files
    n_total_requests: int = 0

    # request files that have not been submitted yet
    unsubmitted_request_files: list[str] = field(default_factory=list)

    # batches in OpenAI
    submitted_batches: dict[str, GenericBatchObject] = field(default_factory=dict)
    finished_batches: dict[str, GenericBatchObject] = field(default_factory=dict)
    downloaded_batches: dict[str, GenericBatchObject] = field(default_factory=dict)

    @property
    def n_total_batches(self) -> int:
        return (
            self.n_unsubmitted_request_files
            + self.n_submitted_batches
            + self.n_finished_batches
            + self.n_downloaded_batches
        )

    @property
    def n_unsubmitted_request_files(self) -> int:
        return len(self.unsubmitted_request_files)

    @property
    def n_submitted_batches(self) -> int:
        return len(self.submitted_batches)

    @property
    def n_finished_batches(self) -> int:
        return len(self.finished_batches)

    @property
    def n_downloaded_batches(self) -> int:
        return len(self.downloaded_batches)

    @property
    def n_finished_requests(self) -> int:
        batches = list(self.submitted_batches.values()) + list(self.finished_batches.values())
        return sum(b.request_counts.completed + b.request_counts.failed for b in batches)

    @property
    def n_downloaded_requests(self) -> int:
        batches = list(self.downloaded_batches.values())
        return sum(b.request_counts.completed + b.request_counts.failed for b in batches)

    @property
    def n_finished_or_downloaded_requests(self) -> int:
        return self.n_finished_requests + self.n_downloaded_requests

    @property
    def n_submitted_finished_or_downloaded_batches(self) -> int:
        return self.n_submitted_batches + self.n_finished_batches + self.n_downloaded_batches

    @property
    def n_finished_or_downloaded_batches(self) -> int:
        return self.n_finished_batches + self.n_downloaded_batches

    def mark_as_submitted(
        self, request_file: str, batch_object: GenericBatchObject, n_requests: int
    ):
        assert request_file in self.unsubmitted_request_files
        assert n_requests > 0
        self.unsubmitted_request_files.remove(request_file)
        self.submitted_batches[batch_object.id] = batch_object
        self.n_total_requests += n_requests
        logger.debug(f"Marked {request_file} as submitted with batch {batch_object.id}")

    def mark_as_finished(self, batch_object: GenericBatchObject):
        assert batch_object.id in self.submitted_batches
        self.submitted_batches.pop(batch_object.id)
        self.finished_batches[batch_object.id] = batch_object
        logger.debug(f"Marked batch {batch_object.id} as finished")

    def mark_as_downloaded(self, batch_object: GenericBatchObject):
        assert batch_object.id in self.finished_batches
        self.finished_batches.pop(batch_object.id)
        self.downloaded_batches[batch_object.id] = batch_object
        logger.debug(f"Marked batch {batch_object.id} as downloaded")

    def update_submitted(self, batch_object: GenericBatchObject):
        assert batch_object.id in self.submitted_batches
        self.submitted_batches[batch_object.id] = batch_object
        logger.debug(f"Updated submitted batch {batch_object.id} with new request counts")

    def __str__(self) -> str:
        """Returns a human-readable string representation of the batch status."""
        status_lines = [
            f"Total batches: {self.n_total_batches}",
            f"Unsubmitted files: {self.n_unsubmitted_request_files}",
            f"Submitted batches: {self.n_submitted_batches}",
            f"Finished batches: {self.n_finished_batches}",
            f"Downloaded batches: {self.n_downloaded_batches}",
            "",
            f"Total requests: {self.n_total_requests}",
            f"Finished requests: {self.n_finished_requests}",
            f"Downloaded requests: {self.n_downloaded_requests}",
        ]
        return "\n".join(status_lines)
