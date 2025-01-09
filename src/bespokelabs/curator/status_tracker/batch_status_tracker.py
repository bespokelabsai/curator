import logging

from pydantic import BaseModel, Field

from bespokelabs.curator.types.generic_batch import GenericBatch, GenericBatchStatus

logger = logging.getLogger(__name__)


class BatchStatusTracker(BaseModel):
    n_total_requests: int = Field(default=0)
    unsubmitted_request_files: set[str] = Field(default_factory=set)
    submitted_batches: dict[str, GenericBatch] = Field(default_factory=dict)
    finished_batches: dict[str, GenericBatch] = Field(default_factory=dict)
    downloaded_batches: dict[str, GenericBatch] = Field(default_factory=dict)

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
        return sum(b.request_counts.succeeded + b.request_counts.failed for b in batches)

    @property
    def n_downloaded_requests(self) -> int:
        batches = list(self.downloaded_batches.values())
        return sum(b.request_counts.succeeded + b.request_counts.failed for b in batches)

    @property
    def n_finished_or_downloaded_requests(self) -> int:
        return self.n_finished_requests + self.n_downloaded_requests

    @property
    def n_submitted_finished_or_downloaded_batches(self) -> int:
        return self.n_submitted_batches + self.n_finished_batches + self.n_downloaded_batches

    @property
    def n_finished_or_downloaded_batches(self) -> int:
        return self.n_finished_batches + self.n_downloaded_batches

    def mark_as_submitted(self, batch: GenericBatch, n_requests: int):
        assert n_requests > 0
        batch.status = GenericBatchStatus.SUBMITTED
        if batch.request_file in self.unsubmitted_request_files:
            self.unsubmitted_request_files.remove(batch.request_file)
            self.n_total_requests += n_requests
        else:
            logger.warning(f"Request file {batch.request_file} has already been submitted.")
        self.submitted_batches[batch.id] = batch
        logger.debug(f"Marked {batch.request_file} as submitted with batch {batch.id}")

    def mark_as_finished(self, batch: GenericBatch):
        assert batch.id in self.submitted_batches
        batch.status = GenericBatchStatus.FINISHED
        self.submitted_batches.pop(batch.id)
        self.finished_batches[batch.id] = batch
        logger.debug(f"Marked batch {batch.id} as finished")

    def mark_as_downloaded(self, batch: GenericBatch):
        assert batch.id in self.finished_batches
        batch.status = GenericBatchStatus.DOWNLOADED
        self.finished_batches.pop(batch.id)
        self.downloaded_batches[batch.id] = batch
        logger.debug(f"Marked batch {batch.id} as downloaded")

    def update_submitted(self, batch: GenericBatch):
        assert batch.id in self.submitted_batches
        self.submitted_batches[batch.id] = batch
        logger.debug(f"Updated submitted batch {batch.id} with new request counts")

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
