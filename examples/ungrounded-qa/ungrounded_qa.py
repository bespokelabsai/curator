"""Generate diverse set of questions and answers by generating diverse subjects and subsubjects.

This is similar to how data is generated for the Camel dataset.
See section F (appendix) of https://arxiv.org/pdf/2303.17760.
"""

from typing import List

from pydantic import BaseModel, Field

from bespokelabs import curator


class Subject(BaseModel):
    """A single subject."""

    subject: str = Field(description="A subject")


class Subjects(BaseModel):
    """A list of subjects."""

    subjects: List[Subject] = Field(description="A list of subjects")


class QA(BaseModel):
    """A question and answer pair."""

    question: str = Field(description="A question")
    answer: str = Field(description="An answer")


class QAs(BaseModel):
    """A list of question and answer pairs."""

    qas: List[QA] = Field(description="A list of QAs")


class SubjectGenerator(curator.LLM):
    """A subject generator that generates diverse subjects."""

    response_format = Subjects

    @classmethod
    def prompt(cls, input: dict) -> str:
        """Generate a prompt for the subject generator."""
        return "Generate a diverse list of 3 subjects. Keep it high-level (e.g. Math, Science)."

    @classmethod
    def parse(cls, input: dict, response: Subjects) -> dict:
        """Parse the model response into the desired output format."""
        return response.subjects


class SubsubjectGenerator(curator.LLM):
    """A subsubject generator that generates diverse subsubjects for a given subject."""

    response_format = Subjects

    @classmethod
    def prompt(cls, input: dict) -> str:
        """Generate a prompt for the subsubject generator."""
        return f"For the given subject {input['subject']}. Generate 3 diverse subsubjects. No explanation."

    @classmethod
    def parse(cls, input: dict, response: Subjects) -> dict:
        """Parse the model response into the desired output format."""
        return [{"subject": input["subject"], "subsubject": subsubject.subject} for subsubject in response.subjects]


class QAGenerator(curator.LLM):
    """A QA generator that generates diverse questions and answers for a given subsubject."""

    response_format = QAs

    @classmethod
    def prompt(cls, input: dict) -> str:
        """Generate a prompt for the QA generator."""
        return f"For the given subsubject {input['subsubject']}. Generate 3 diverse questions and answers. No explanation."

    @classmethod
    def parse(cls, input: dict, response: QAs) -> dict:
        """Parse the model response into the desired output format."""
        return [
            {
                "subject": input["subject"],
                "subsubject": input["subsubject"],
                "question": qa.question,
                "answer": qa.answer,
            }
            for qa in response.qas
        ]


def main():
    """Main function to generate a dataset of questions and answers."""
    subject_generator = SubjectGenerator(model_name="gpt-4o-mini")
    subject_dataset = subject_generator()

    subsubject_generator = SubsubjectGenerator(model_name="gpt-4o-mini")
    subsubject_dataset = subsubject_generator(subject_dataset)

    qa_generator = QAGenerator(model_name="gpt-4o-mini")
    qa_dataset = qa_generator(subsubject_dataset)
    qa_dataset = qa_dataset.map(lambda row: {"answer": row["answer"].strip()}, num_proc=2)
    print(qa_dataset.to_pandas())


if __name__ == "__main__":
    main()
