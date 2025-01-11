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


def create_camel(temp_working_dir, batch=False):
    if batch:
        backend_params = {"batch_check_interval": batch_check_interval}
    else:
        backend_params = {}

    subject_prompter = curator.LLM(
        prompt_func=lambda: "Generate a diverse list of 3 subjects. Keep it high-level (e.g. Math, Science).",
        parse_func=lambda _, subjects: list(subjects.subjects),
        model_name="gpt-4o-mini",
        response_format=Subjects,
        backend_params=backend_params,
    )
    subsubject_prompter = curator.LLM(
        prompt_func=lambda subject: f"For the given subject {subject}. Generate 3 diverse subsubjects. No explanation.",
        parse_func=lambda subject, subsubjects: [{"subject": subject["subject"], "subsubject": subsubject.subject} for subsubject in subsubjects.subjects],
        model_name="gpt-4o-mini",
        response_format=Subjects,
        backend_params=backend_params,
    )

    qa_prompter = curator.LLM(
        prompt_func=lambda subsubject: f"For the given subsubject {subsubject}. Generate 3 diverse questions and answers. No explanation.",
        model_name="gpt-4o-mini",
        response_format=QAs,
        parse_func=lambda subsubject, qas: [
            {
                "subject": subsubject["subject"],
                "subsubject": subsubject["subsubject"],
                "question": qa.question,
                "answer": qa.answer,
            }
            for qa in qas.qas
        ],
    )

    subject_dataset = subject_prompter()
    subsubject_dataset = subsubject_prompter(subject_dataset)
    qa_dataset = qa_prompter(subsubject_dataset, working_dir=temp_working_dir)
    qa_dataset = qa_dataset.map(lambda row: {"answer": row["answer"].strip()}, num_proc=2)
    return qa_dataset


def prompt_func(row):
    return row["dish"]


def parse_func(row, response):
    return {"recipe": response}


_DEFAULT_MODEL_MAP = {
    "openai": "gpt-3.5-turbo",
    "anthropic": "claude-3-5-sonnet-20241022",
    "litellm": "gpt-3.5-turbo",
    "vllm": "Qwen/Qwen2.5-1.5B-Instruct",
}


def create_basic(temp_working_dir, mock_dataset, llm_params=None, batch=False, backend="openai", mocking=None, batch_cancel=False, tracker_console=None):
    from bespokelabs import curator

    llm_params = llm_params or {}
    if batch:
        llm_params["batch_check_interval"] = batch_check_interval

    prompter = curator.LLM(
        prompt_func=prompt_func,
        parse_func=parse_func,
        model_name=_DEFAULT_MODEL_MAP[backend],
        backend=backend,
        batch=batch,
        backend_params=llm_params,
    )
    prompter._request_processor._tracker_console = tracker_console
    if mocking:
        prompter = mocking(prompter)
    if batch:
        prompter._hash_fingerprint = lambda x, y: "testing_hash_123"
    dataset = prompter(mock_dataset, working_dir=temp_working_dir, batch_cancel=batch_cancel)
    return dataset


def create_llm(batch=False):
    if batch:
        backend_params = {"batch_check_interval": batch_check_interval}
    else:
        backend_params = {}

    prompter = curator.LLM(
        prompt_func=prompt_func,
        parse_func=parse_func,
        model_name="gpt-3.5-turbo",
        backend="openai",
        backend_params=backend_params,
    )
    return prompter
