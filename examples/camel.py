from typing import List

import pandas as pd
from pydantic import BaseModel, Field

import bella
import prompt


class Subjects(BaseModel):
    subjects: List[str] = Field(description="A list of subjects")


class QA(BaseModel):
    question: str = Field(description="A question")
    answer: str = Field(description="A answer")


class QAs(BaseModel):
    qas: List[QA] = Field(description="A list of QAs")

@bella.completion(
    model="gpt-4o-mini",
    response_model=Subjects,
)
def generate_subjects(n: int = 3):
    return f"Generate a diverse list of {n} subjects. Keep it high-level (e.g. Math, Science)."

@bella.completion(model="gpt-4o-mini", response_model=Subjects)
def generate_subsubjects(subject: str) -> str:
    return f"For the subject {subject}, generate 3 diverse subsubjects within the subject."

@bella.completion(
    model="gpt-4o-mini",
    response_model=QAs,
)
def generate_qas(subject: str):
    """You are a helpful AI assistant."""
    return f"For the given subject {subject}, generate 3 questions and answers."

subjects = generate_subjects()
subjects = [subject for subject in subjects.subjects]

subsubjects = bella.parallel(generate_subsubjects)(subjects)
flattened_subsubjects = [
    {"subject": subject, "subsubject": subsubject} 
    for subject, subsubjects_list in zip(subjects, subsubjects)
    for subsubject in subsubjects_list.subjects
]

qas = bella.parallel(generate_qas)(flattened_subsubjects)
flattened_qas = [
    {"subject": subsubject["subject"], "subsubject": subsubject["subsubject"], "question": qa.question, "answer": qa.answer}
    for subsubject, qas_list in zip(flattened_subsubjects, qas)
    for qa in qas_list.qas
]

qas_df = pd.DataFrame(flattened_qas)
print(qas_df)
# camelai()
