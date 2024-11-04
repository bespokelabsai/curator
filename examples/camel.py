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


GetSubjects = prompt.Prompter(
    system_prompt="You are a helpful AI assistant.",
    user_prompt="Generate a diverse list of 3 subjects. Keep it high-level (e.g. Math, Science).",
    response_format=Subjects,
    model_name="gpt-4o-mini",
)


GetSubSubjects = prompt.Prompter(
    system_prompt="You are a helpful AI assistant.",
    user_prompt="For the given subject {{ subject }}. Generate 3 diverse subsubjects. No explanation.",
    response_format=Subjects,
    model_name="gpt-4o-mini",
)


GetQAList = prompt.Prompter(
    system_prompt="You are a helpful AI assistant.",
    user_prompt="For the given subject {{ subsubject }}, generate 1 diverse questions and answers. No explanation.",
    response_format=QAs,
    model_name="gpt-4o-mini",
)


def camelai():
    subject_dataset = bella.completions(
        prompter=GetSubjects,
        name="Generate subjects",
    )
    # If the output is a list, bella automatically flattens it.
    subject_dataset = bella.map(
        subject_dataset,
        # {"subjects": ["Math", "Science"]} -> [{"subject": "Math"}, {"subject": "Science"}]
        lambda sample: [{"subject": subject} for subject in sample["subjects"]],
    )
    print(pd.DataFrame.from_records(subject_dataset))

    subsubject_dataset = bella.completions(
        dataset=subject_dataset,
        prompter=GetSubSubjects,
        name="Generate subsubjects",
    )
    # join the subject and subsubject datasets
    subsubject_dataset = bella.map(
        zip(subject_dataset, subsubject_dataset),
        # ({"subject": "Math"}, {"subjects": ["Algebra", "Geometry"]}) -> [{"subject": "Math", "subsubject": "Algebra"}, {"subject": "Math", "subsubject": "Geometry"}]
        lambda sample: [
            {"subject": sample[0]["subject"], "subsubject": subsubject}
            for subsubject in sample[1]["subjects"]
        ],
    )
    print(pd.DataFrame.from_records(subsubject_dataset))

    qa_dataset = bella.completions(
        subsubject_dataset,
        prompter=GetQAList,
        name="Generate QAs",
    )
    qa_dataset = bella.map(
        zip(subsubject_dataset, qa_dataset),
        lambda sample: [
            {
                "subject": sample[0]["subject"],
                "subsubject": sample[0]["subsubject"],
                "question": qa["question"],
                "answer": qa["answer"],
            }
            for qa in sample[1]["qas"]
        ],
    )
    print(pd.DataFrame.from_records(qa_dataset))


camelai()
