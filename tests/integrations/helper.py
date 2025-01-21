import typing as t

from pydantic import BaseModel, Field

from bespokelabs import curator

batch_check_interval = 1


class Subject(BaseModel):
    subject: str = Field(description="A subject")


class Subjects(BaseModel):
    subjects: t.List[Subject] = Field(description="A list of subjects")


class QA(BaseModel):
    question: str = Field(description="A question")
    answer: str = Field(description="An answer")


class QAs(BaseModel):
    qas: t.List[QA] = Field(description="A list of QAs")


class BasicLLM(curator.LLM):
    def prompt(self, input: dict) -> str:
        return input["dish"]

    def parse(self, input: dict, response) -> dict:
        return {"recipe": response}


class Poems(BaseModel):
    poems_list: t.List[str] = Field(description="A list of poems.")


class Poet(curator.LLM):
    response_format = Poems

    def prompt(self, input: dict) -> str:
        return "Write two simple poems."

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


def create_camel(temp_working_dir, batch=False):
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
}


def create_basic(
    temp_working_dir, mock_dataset, llm_params=None, batch=False, backend="openai", mocking=None, batch_cancel=False, tracker_console=None, model=None
):
    llm_params = llm_params or {}
    if batch:
        llm_params["batch_check_interval"] = batch_check_interval

    if mock_dataset is None:
        prompter = Poet(
            model_name=model or _DEFAULT_MODEL_MAP[backend],
            backend=backend,
            batch=batch,
            backend_params=llm_params,
        )
    else:
        prompter = BasicLLM(
            model_name=model or _DEFAULT_MODEL_MAP[backend],
            backend=backend,
            batch=batch,
            backend_params=llm_params,
        )
    prompter._request_processor._tracker_console = tracker_console
    if mocking:
        prompter = mocking(prompter)
    if batch:
        prompter._hash_fingerprint = lambda x, y: "testing_hash_123"
    if mock_dataset:
        dataset = prompter(mock_dataset, working_dir=temp_working_dir, batch_cancel=batch_cancel)
    else:
        dataset = prompter(working_dir=temp_working_dir, batch_cancel=batch_cancel)
    return dataset


def create_llm(batch=False):
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
