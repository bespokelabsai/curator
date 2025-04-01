# ruff: noqa: E731
"""This module contains a pipeline for generating stratified questions.

# Reference:
https://arxiv.org/abs/2410.09038
"""

import random
from typing import List

from datasets import Dataset
from pydantic import BaseModel, Field

from bespokelabs import curator


class PropertyList(BaseModel):
    """A list of properties for potential answers to a question."""

    properties_list: List[str] = Field(description="List of true/false properties for potential answers")


class PropertyEvaluation(BaseModel):
    """Evaluation of a property's probability of being true for a random answer."""

    strata: str = Field(description="The property being evaluated")
    probability: float = Field(description="Probability (0-1) that the property is true for a random answer")
    reasoning: str = Field(description="Reasoning behind the probability estimate")


class OptimalProperties(BaseModel):
    """Selection of optimal properties for answer identification."""

    selected_properties: List[PropertyEvaluation] = Field(description="List of optimal properties for answer identification")


_AutoStratificationPrompt = """I am tasked with the following request:
        {user_request}
Help me brainstorm how to respond to the user request by providing a list of True/False properties the solution may or may not have.
Use the following step-by-step to come up with good properties:
1. If you were playing 20 questions, what’s a good first question to ask that would split the possibilities in half?
List at least 5 questions and their corresponding properties.
Question: <Description>
2. Rewrite each question as a True/False property that’s true for one half and false for the other.
Question: <Description>
True/False Property: <Property Description>
3. For each property, come up with an example that would satisfy the property.
Property: <Description>
Example: <Description>
Is it a valid answer to the user’s request? <Yes/No>
4. For each property, come up with an example that would not satisfy the property.
Property: <Description>
Example: <Description>
Is it a valid answer to the user’s request? <Yes/No>
5. Does the property mention a candidate answer in it?
Property: <Description>
Does the property mention a candidate answer in it? <Yes/No>
6. For each property, list whether we should include it or not in the final list of properties.
Do not include ones where an example from above is not valid or if it mentions a candidate answer in it.
Property: <Description>
Include in final list? <Yes/No>
Final List of True/False Properties:
1. <Property Description 1>
2. <Property Description 2>
Ensure all properties are listed are sentences that are either True or False
 """

_HeuristicEstimationPrompt = (
    lambda q, p: f"""I am tasked to estimate the probability that a random solution to "{q}" has the following property "{p}"

Instructions:
1. Provide at least 3 reasons why the answer might be no.
{{ Insert your thoughts }}
2. Provide at least 3 reasons why the answer might be yes.
{{ Insert your thoughts }}
3. Rate the strength of each of the reasons given in the last two responses. Think like a superforecaster (e.g. Nate Silver).
{{ Insert your rating of the strength of each reason }}
4. Aggregate your considerations.
{{ Insert your aggregated considerations }}
5. Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
{{ Insert your answer }}"""
)

_HeuristicEstimationPromptOptimal = """I’m playing a game where my friend has been tasked to:

"{question}"

I have the following Y/N statements I can ask my friend. I have probabilities that I think it’s true: {properties}
Instructions:
1. For each Y/N statement, is it redundant with another statement?
Y/N statement: <description>
Is redundant? <Y/N: Explanation>
2. Are any of the probabilities in accurate? If it’s sufficiently accurate just report back the same value.
Y/N statement: <Description>
Is accurate? <Y/N: Explanation>
Probability: <Probability>
3. Pick at most three statements that are least redundant and pair well together. Prefer ones that are closest to 50% for most information.
Final List of True/False Properties:
1. <Y/N Properties> :: <Probability>
2. <Y/N Properties> :: <Probability>
"""


class AutoStratification(curator.LLM):
    """Generator for true/false properties for potential answers."""

    response_format = PropertyList

    def prompt(self, input: dict) -> str:
        """Generate a prompt for property generation."""
        question = input.get("question", "Name a US state?")
        return [
            {"role": "user", "content": _AutoStratificationPrompt.format(user_request=question)},
            {"role": "system", "content": "You’re a helpful brainstorming assistant that is careful to consider all factors to a problem."},
        ]

    def parse(self, input: dict, response: PropertyList) -> dict:
        """Parse the model response into the desired output format."""
        return [{"question": input["question"], "property": property} for property in response.properties_list]


class HeuristicEstimation(curator.LLM):
    """Generator for estimating probability for each properties."""

    response_format = PropertyEvaluation

    def prompt(self, input: dict) -> str:
        """Generate a prompt for property generation."""
        return [
            {"role": "user", "content": _HeuristicEstimationPrompt(input["question"], input["property"])},
            {
                "role": "system",
                "content": "You are an expert superforecaster, familiar with the work of Tetlock and others. Your mission is to generate accurate predictions for forecasting questions. Aggregate the information provided by the user. Make sure to give detailed reasoning.",  # noqa
            },
        ]

    def parse(self, input: dict, response: PropertyList) -> dict:
        """Parse the model response into the desired output format."""
        return {"question": input["question"], "property": input["property"], "probability": response.probability, "reasoning": response.reasoning}

    def collate(self, out):
        """Collate the output into a single dataset."""
        df = out.to_pandas()
        grouped_df = df.groupby("question").agg({"property": list, "probability": list, "reasoning": lambda x: " | ".join(x)}).reset_index()
        return Dataset.from_pandas(grouped_df)


class HeuristicEstimationNegation(curator.LLM):
    """Generator for redundant properties and correcting the probabltities."""

    response_format = OptimalProperties

    def prompt(self, input: dict) -> str:
        """Generate a prompt for property generation."""
        properties = [(pr, p) for pr, p in zip(input["property"], input["probability"])]
        return [
            {"role": "user", "content": _HeuristicEstimationPromptOptimal.format(question=input["question"], properties=properties)},
            {
                "role": "system",
                "content": "You are an expert superforecaster, familiar with the work of Tetlock and others. Your mission is to generate accurate predictions for forecasting questions. Aggregate the information provided by the user. Make sure to give detailed reasoning.",  # noqa
            },
        ]

    def parse(self, input: dict, response: OptimalProperties) -> dict:
        """Parse the model response into the desired output format."""
        properties = [p.strata for p in response.selected_properties]
        probabilities = [p.probability for p in response.selected_properties]
        return {"question": input["question"], "properties": properties, "probabilities": probabilities}


class StratifiedQA(curator.LLM):
    """Generator for true/false properties for potential answers."""

    def prompt(self, input: dict) -> str:
        """Generate a prompt for property generation."""
        properties = input["properties"]
        weights = input["probabilities"]
        sampled_property = random.choices(properties, weights=weights, k=1)[0]

        return f'{sampled_property} {input["question"]}'

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response into the desired output format."""
        return {"question": input["question"], "answer": response}


class StratifiedGenerator:
    """Stratified question generation pipeline."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the pipeline with the necessary models."""
        self.args = args
        self.kwargs = kwargs
        self.autostrat = AutoStratification(*self.args, **self.kwargs)
        self.heuristic = HeuristicEstimation(*self.args, **self.kwargs)
        self.heuristic_negation = HeuristicEstimationNegation(*self.args, **self.kwargs)
        self.qa = StratifiedQA(*self.args, **self.kwargs)

    def __call__(self, questions: Dataset, working_dir: str | None = None) -> Dataset:
        """Generate questions for a given input."""
        out1 = self.autostrat(questions, working_dir=working_dir)
        out2 = self.heuristic(out1, working_dir=working_dir)
        out2 = self.heuristic.collate(out2)
        out2_2 = self.heuristic_negation(out2, working_dir=working_dir)
        qas = self.qa(out2_2, working_dir=working_dir)
        return qas
