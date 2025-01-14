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
    subject_prompter = curator.LLM(
        prompt_func=lambda: "Generate a diverse list of 3 subjects. Keep it high-level (e.g. Math, Science).",
        parse_func=lambda _, subjects: list(subjects.subjects),
        model_name="gpt-4o-mini",
        response_format=Subjects,
        batch_check_interval=batch_check_interval,
    )
    subsubject_prompter = curator.LLM(
        prompt_func=lambda subject: f"For the given subject {subject}. Generate 3 diverse subsubjects. No explanation.",
        parse_func=lambda subject, subsubjects: [{"subject": subject["subject"], "subsubject": subsubject.subject} for subsubject in subsubjects.subjects],
        model_name="gpt-4o-mini",
        response_format=Subjects,
        batch_check_interval=batch_check_interval,
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
        batch_check_interval=batch_check_interval,
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


def create_basic(temp_working_dir, mock_dataset, llm_params=None, batch=False, backend="openai", mocking=None, batch_cancel=False):
    llm_params = llm_params or {}
    prompter = curator.LLM(
        prompt_func=prompt_func,
        parse_func=parse_func,
        model_name="gpt-3.5-turbo",
        backend=backend,
        batch_check_interval=batch_check_interval,
        batch=batch,
        **llm_params,
    )
    if mocking:
        prompter = mocking(prompter)
    if batch:
        prompter._hash_fingerprint = lambda x: "testing_hash_123"
    dataset = prompter(mock_dataset, working_dir=temp_working_dir, batch_cancel=batch_cancel)
    return dataset


def create_llm(batch=False):
    prompter = curator.LLM(
        prompt_func=prompt_func,
        parse_func=parse_func,
        model_name="gpt-3.5-turbo",
        backend="openai",
        batch_check_interval=batch_check_interval,
    )
    return prompter
