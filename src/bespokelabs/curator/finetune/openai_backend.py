import os

from openai import OpenAI

from bespokelabs.curator.finetune.base_backend import BaseFinetuneBackend


class OpenAIFinetuneBackend(BaseFinetuneBackend):
    """A client for interacting with the OpenAI Finetuning API.

    This class provides methods to create, list, and manage finetuning jobs
    on the OpenAI platform. It serves as a backend for the Finetune class.
    """

    def __init__(self, backend_params: dict):
        """Initializes the OpenAIFinetuneBackend.

        Args:
            backend_params: A dictionary containing configuration for the backend.
                            Currently, this is not used by the OpenAI backend but is
                            part of the BaseFinetuneBackend interface.
                            The OpenAI API key is retrieved from the environment variable
                            "OPENAI_API_KEY".
        """
        super().__init__(backend_params)
        # Initialize the OpenAI client using the API key from environment variables.
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def create_job(self, **kwargs):
        """Creates a new finetuning job on the OpenAI platform.

        Args:
            **kwargs: Arbitrary keyword arguments passed directly to the
                      `openai.fine_tuning.jobs.create` method. Expected keys include:
                      'dataset_id' (str): The ID of an uploaded file that contains training data.
                      'model_name' (str): The name of the model to fine-tune.
                      'method' (Optional[dict]): The finetuning method and its hyperparameters.
                                                 (Note: OpenAI API might use 'hyperparameters' directly
                                                 or a nested structure depending on the API version and model type).
                      'hyperparameters' (Optional[dict]): The hyperparameters used for the fine-tuning job.
                                                          (This might be deprecated in favor of 'method' in some contexts).
                      'seed' (Optional[int]): The seed to use for reproducibility.
                      'suffix' (Optional[str]): A string of up to 40 characters that will be added to your
                                                fine-tuned model name.
                      'validation_file' (Optional[str]): The ID of an uploaded file that contains validation data.
                      Refer to the OpenAI API documentation for a complete list of parameters.

        Returns:
            openai.types.fine_tuning.FineTuningJob: An object representing the created fine-tuning job.
        """
        job = self.client.fine_tuning.jobs.create(
            training_file=kwargs["dataset_id"],
            model=kwargs["model_name"],
            hyperparameters=kwargs.get("hyperparameters"),
            seed=kwargs.get("seed"),
            suffix=kwargs.get("suffix"),
            validation_file=kwargs.get("validation_file"),
        )
        return job

    def list_jobs(self, **kwargs):
        """Lists all finetuning jobs for the organization on the OpenAI platform.

        Args:
            **kwargs: Arbitrary keyword arguments passed directly to the
                      `openai.fine_tuning.jobs.list` method. Common keys include:
                      'after' (Optional[str]): Identifier for the last job from the previous pagination request.
                      'limit' (Optional[int]): Number of fine-tuning jobs to retrieve.
                      Refer to the OpenAI API documentation for a complete list of parameters.

        Returns:
            openai.pagination.SyncCursorPage[openai.types.fine_tuning.FineTuningJob]:
                A paginated list of fine-tuning jobs.
        """
        # List fine-tuning jobs using the OpenAI client.
        # All provided keyword arguments are passed to the OpenAI API.
        jobs = self.client.fine_tuning.jobs.list(**kwargs)
        return jobs

    def list_job_events(self, **kwargs):
        """Lists events for a specific finetuning job on the OpenAI platform.

        Args:
            **kwargs: Arbitrary keyword arguments. Expected key:
                      'job_id' (str): The ID of the fine-tuning job to retrieve events for.
                                      This is passed as `fine_tuning_job_id` to the OpenAI API.
                      Other kwargs are passed directly to `openai.fine_tuning.jobs.list_events`.
                      Common keys include:
                      'after' (Optional[str]): Identifier for the last event from the previous pagination request.
                      'limit' (Optional[int]): Number of events to retrieve.
                      Refer to the OpenAI API documentation for a complete list of parameters.

        Returns:
            openai.pagination.SyncCursorPage[openai.types.fine_tuning.FineTuningJobEvent]:
                A paginated list of fine-tuning job events.
        """
        # Retrieve the job_id from kwargs and remove it to prevent it from being passed again.
        fine_tuning_job_id = kwargs.pop("job_id")
        # List events for a specific fine-tuning job using the OpenAI client.
        events = self.client.fine_tuning.jobs.list_events(fine_tuning_job_id=fine_tuning_job_id, **kwargs)
        return events

    def get_job_details(self, **kwargs):
        """Retrieves details for a specific finetuning job on the OpenAI platform.

        Args:
            **kwargs: Arbitrary keyword arguments. Expected key:
                      'job_id' (str): The ID of the fine-tuning job to retrieve.
                                      This is passed as `fine_tuning_job_id` to the OpenAI API.
                      Other kwargs are passed directly to `openai.fine_tuning.jobs.retrieve`.
                      Refer to the OpenAI API documentation for a complete list of parameters.

        Returns:
            openai.types.fine_tuning.FineTuningJob: An object representing the fine-tuning job details.
        """
        # Retrieve the job_id from kwargs and remove it to prevent it from being passed again.
        fine_tuning_job_id = kwargs.pop("job_id")
        # Retrieve details for a specific fine-tuning job using the OpenAI client.
        job = self.client.fine_tuning.jobs.retrieve(fine_tuning_job_id=fine_tuning_job_id, **kwargs)
        return job

    def cancel_job(self, **kwargs):
        """Cancels a specific finetuning job on the OpenAI platform.

        Args:
            **kwargs: Arbitrary keyword arguments. Expected key:
                      'job_id' (str): The ID of the fine-tuning job to cancel.
                                      This is passed as `fine_tuning_job_id` to the OpenAI API.
                      Other kwargs are passed directly to `openai.fine_tuning.jobs.cancel`.
                      Refer to the OpenAI API documentation for a complete list of parameters.

        Returns:
            openai.types.fine_tuning.FineTuningJob: An object representing the cancelled fine-tuning job.
        """
        # Retrieve the job_id from kwargs and remove it to prevent it from being passed again.
        fine_tuning_job_id = kwargs.pop("job_id")
        # Cancel a specific fine-tuning job using the OpenAI client.
        job = self.client.fine_tuning.jobs.cancel(fine_tuning_job_id=fine_tuning_job_id, **kwargs)
        return job
