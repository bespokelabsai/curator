import typing as t

from pydantic import BaseModel, Field

from bespokelabs import curator


class Subject(BaseModel):
    subject: str = Field(description="A subject")


class Subjects(BaseModel):
    subjects: t.List[Subject] = Field(description="A list of subjects")


class QA(BaseModel):
    question: str = Field(description="A question")
    answer: str = Field(description="An answer")


class QAs(BaseModel):
    qas: t.List[QA] = Field(description="A list of QAs")


class Reasoner(curator.LLM):
    return_completions_object = True

    def prompt(self, input):  # noqa: D102
        return input["question"]

    def parse(self, input, response):
        """Parse the LLM response to extract reasoning and solution."""
        content = response["content"]
        thinking = ""
        text = ""

        for content_block in content:
            if content_block["type"] == "thinking":
                if "thinking" in content_block:
                    thinking = content_block["thinking"]
                else:
                    thinking = content_block["text"]

            elif content_block["type"] == "text":
                text = content_block["text"]
            elif content_block["type"] == "redacted_thinking":
                print("Redacted thinking block! (notifying you for fun)")

        input["model_thinking_trajectory"] = thinking
        input["model_attempt"] = text
        return input


class BasicLLM(curator.LLM):
    def __init__(self, *args, **kwargs):
        self.raw_prompt = kwargs.pop("raw_prompt", False)
        super().__init__(*args, **kwargs)

    def prompt(self, input: dict) -> str:
        if self.raw_prompt:
            return [{"role": "user", "content": input["text"]}]
        return input["dish"]

    def parse(self, input: dict, response) -> dict:
        return {"recipe": response}


class MultiModalLLM(curator.LLM):
    def __init__(self, *args, **kwargs):
        self.input_type = kwargs.pop("input_type", "image")
        super().__init__(*args, **kwargs)

    def prompt(self, input: dict):
        if self.input_type == "image":
            if isinstance(input["image"], str):
                return input["text"], curator.types.Image(url=input["image"])
            return input["text"], curator.types.Image(content=input["image"])
        else:
            return input["text"], curator.types.File(url=input["pdf"])

    def parse(self, input: dict, response) -> dict:
        return {"description": response}


class Poems(BaseModel):
    poems_list: t.List[str] = Field(description="A list of poems.")


class Poet(curator.LLM):
    response_format = Poems

    def __init__(self, *args, **kwargs):
        self.raw_prompt = kwargs.pop("raw_prompt", False)
        super().__init__(*args, **kwargs)

    def prompt(self, input: dict) -> str:
        prompt = "Write two simple poems."
        if self.raw_prompt:
            return [{"role": "user", "content": prompt}]
        return prompt

    def parse(self, input: dict, response: Poems) -> dict:
        return [{"poem": p} for p in response.poems_list]


class SubjectLLM(curator.LLM):
    response_format = Subjects

    def prompt(self, input: dict) -> str:
        return "Generate a diverse list of 3 subjects. Keep it high-level (e.g. Math, Science)."

    def parse(self, input: dict, response) -> dict:
        return list(response.subjects)


class SubsubjectLLM(curator.LLM):
    response_format = Subjects

    def prompt(self, input: dict) -> str:
        return f"For the given subject {input['subject']}. Generate 3 diverse subsubjects. No explanation."

    def parse(self, input: dict, response) -> dict:
        return [{"subject": input["subject"], "subsubject": subsubject.subject} for subsubject in response.subjects]


class QALLM(curator.LLM):
    response_format = QAs

    def prompt(self, input: dict) -> str:
        return f"For the given subsubject {input['subsubject']}. Generate 3 diverse questions and answers. No explanation."

    def parse(self, input: dict, response) -> dict:
        return [
            {
                "subject": input["subject"],
                "subsubject": input["subsubject"],
                "question": qa.question,
                "answer": qa.answer,
            }
            for qa in response.qas
        ]


def create_camel(temp_working_dir, batch=False, batch_check_interval=1):
    if batch:
        backend_params = {"batch_check_interval": batch_check_interval}
    else:
        backend_params = {}

    subject_prompter = SubjectLLM(
        model_name="gpt-4o-mini",
        backend_params=backend_params,
    )
    subsubject_prompter = SubsubjectLLM(
        model_name="gpt-4o-mini",
        backend_params=backend_params,
    )

    qa_prompter = QALLM(
        model_name="gpt-4o-mini",
        backend_params=backend_params,
    )

    subject_dataset = subject_prompter()
    subsubject_dataset = subsubject_prompter(subject_dataset)
    qa_dataset = qa_prompter(subsubject_dataset, working_dir=temp_working_dir)
    qa_dataset = qa_dataset.map(lambda row: {"answer": row["answer"].strip()}, num_proc=2)
    return qa_dataset


_DEFAULT_MODEL_MAP = {
    "openai": "gpt-3.5-turbo",
    "anthropic": "claude-3-5-sonnet-20241022",
    "litellm": "gpt-3.5-turbo",
    "vllm": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemini": "gemini-1.5-flash-002",
}


def create_basic(
    temp_working_dir,
    mock_dataset,
    llm_params=None,
    batch=False,
    backend="openai",
    mocking=None,
    batch_cancel=False,
    tracker_console=None,
    model=None,
    return_prompter=False,
    batch_check_interval=1,
    raw_prompt=False,
    generation_params=None,
    parse_func=None,
):
    llm_params = llm_params or {}
    generation_params = generation_params or []
    if batch:
        llm_params["batch_check_interval"] = batch_check_interval

    if mock_dataset is None:
        prompter = Poet(
            model_name=model or _DEFAULT_MODEL_MAP[backend],
            backend=backend,
            batch=batch,
            backend_params=llm_params,
            raw_prompt=raw_prompt,
        )
    elif model and "claude" in model and "3-7" in model:
        prompter = Reasoner(model_name=model, generation_params=generation_params, batch=batch, backend_params=llm_params)
    else:
        prompter = BasicLLM(
            model_name=model or _DEFAULT_MODEL_MAP[backend],
            backend=backend,
            batch=batch,
            backend_params=llm_params,
            raw_prompt=raw_prompt,
        )
    if parse_func:
        prompter.prompt_formatter.parse_func = parse_func
    prompter._request_processor._tracker_console = tracker_console
    if mocking:
        prompter = mocking(prompter)
    if batch:
        prompter._hash_fingerprint = lambda x, y: "testing_hash_123"
    if mock_dataset:
        dataset = prompter(mock_dataset, working_dir=temp_working_dir, batch_cancel=batch_cancel, batch_cancel_auto_confirm=True)
    else:
        dataset = prompter(working_dir=temp_working_dir, batch_cancel=batch_cancel, batch_cancel_auto_confirm=True)
    if return_prompter:
        return dataset.dataset, prompter
    return dataset.dataset


def create_llm(batch=False, batch_check_interval=1):
    if batch:
        backend_params = {"batch_check_interval": batch_check_interval}
    else:
        backend_params = {}

    prompter = BasicLLM(
        model_name="gpt-3.5-turbo",
        backend="openai",
        backend_params=backend_params,
        batch=batch,
    )
    return prompter


def create_multimodal_llm(batch=False, batch_check_interval=1, model="gpt-4o-mini", backend="openai", input_type="image"):
    if batch:
        backend_params = {"batch_check_interval": batch_check_interval}
    else:
        backend_params = {}
    prompter = MultiModalLLM(
        model_name=model,
        backend=backend,
        backend_params=backend_params,
        batch=batch,
        input_type=input_type,
    )
    return prompter


def assert_all_requests_played(cassette, exclude_litellm_cost_request=True):
    played_keys = set(cassette.play_counts.keys())
    unplayed_indices = [i for i in range(len(cassette.requests)) if i not in played_keys]
    # Exclude LiteLLM calls by default since these are flaky.
    if exclude_litellm_cost_request:
        unplayed_indices = [
            i
            for i in unplayed_indices
            if "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json" not in cassette.requests[i].uri
        ]

    if len(unplayed_indices) > 0:
        error_str = "The following requests aren't played: "
        for unplayed_idx in unplayed_indices:
            error_str += f"index: {unplayed_idx}, request_uri: {cassette.requests[unplayed_idx].uri}."
        assert cassette.all_played, f"All requests must be replayed to pass the test. {error_str}"
