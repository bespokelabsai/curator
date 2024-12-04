from typing import List
from pydantic import BaseModel, Field
from bespokelabs import curator


class Subject(BaseModel):
    subject: str = Field(description="A subject")


class Subjects(BaseModel):
    subjects: List[Subject] = Field(description="A list of subjects")


class QA(BaseModel):
    question: str = Field(description="A question")
    answer: str = Field(description="An answer")


class QAs(BaseModel):
    qas: List[QA] = Field(description="A list of QAs")


subject_prompter = curator.Prompter(
    prompt_func=lambda: f"Generate a diverse list of 3 subjects. Keep it high-level (e.g. Math, Science).",
    parse_func=lambda _, subjects: [subject for subject in subjects.subjects],
    model_name="gpt-4o-mini",
    response_format=Subjects,
)
subject_dataset = subject_prompter()
subsubject_prompter = curator.Prompter(
    prompt_func=lambda subject: f"For the given subject {subject}. Generate 3 diverse subsubjects. No explanation.",
    parse_func=lambda subject, subsubjects: [
        {"subject": subject["subject"], "subsubject": subsubject.subject}
        for subsubject in subsubjects.subjects
    ],
    model_name="gpt-4o-mini",
    response_format=Subjects,
)
subsubject_dataset = subsubject_prompter(subject_dataset)

qa_prompter = curator.Prompter(
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
qa_dataset = qa_prompter(subsubject_dataset)

qa_dataset.map(lambda row: {"answer": row["answer"].strip()}, num_proc=2)
print(qa_dataset.to_pandas())
