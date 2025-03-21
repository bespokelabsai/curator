"""RAFT: Adapting Language Model to Domain-Specific RAG.

Reference:
- Paper: https://arxiv.org/html/2403.10131v1
- Reference Code:  https://github.com/ShishirPatil/gorilla/blob/main/raft/raft.py
"""

import functools
import random
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, TypeVar

import datasets
from pydantic import BaseModel, Field

from bespokelabs import curator

ChunkId = int
Content = str
Question = str
ChunkDict = Dict[ChunkId, Dict[str, Content]]
T = TypeVar("T")

_DEFAULT_QUESTION_PROMPT = lambda x, chunk: [  # noqa: E731
    {
        "role": "system",
        "content": """You are a synthetic question-answer pair generator. Given a chunk of context about
             some topic(s), generate %s example questions a user could ask and would be answered using information from the chunk.
             For example, if the given context was a Wikipedia paragraph about the United States, an example question could be
             'How many states are in the United States?'"""
        % (x),
    },
    {"role": "system", "content": "The questions should be able to be answered in a few words or less. Include only the questions in your response."},
    {"role": "user", "content": str(chunk)},
]

_DEFAULT_ANSWER_PROMPT = """
        Question: {question}\nContext: {context}\n
        Answer this question using the information given in the context above. Here is things to pay attention to:
        - First provide step-by-step reasoning on how to answer the question.
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##.
        This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context.
        - End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct.
        You MUST begin your final answer with the tag "<ANSWER>:".
    """


class _Questions(BaseModel):
    """A list of questions."""

    questions: List[str] = Field(description="A list of questions.")


class _RaftQuestion(curator.LLM):
    response_format = _Questions

    def __init__(self, *args, n: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n

    def prompt(self, input: dict) -> str:
        return _DEFAULT_QUESTION_PROMPT(self.n, input["content"])

    def parse(self, input: dict, response: _Questions) -> List[Dict[str, str]]:
        return [{"chunk_id": input["chunk_id"], "question": q} for q in response.questions]


class _SamplingStrategy(Protocol):
    def __call__(self, population: List[T], k: int) -> List[T]: ...


DocumentSet = namedtuple("DocumentSet", ["documents", "oracle_index", "oracle_present"])


@dataclass
class _ContextFormatter:
    document_tag: str = "DOCUMENT"

    def __call__(self, documents: List[str]) -> str:
        return "".join(f"<{self.document_tag}>{doc}</{self.document_tag}>\n" for doc in documents)


class _RaftAnswer(curator.LLM):
    """Enhanced answer generator component for RAFT."""

    def __init__(
        self, chunks: datasets.Dataset, *args, n: int = 5, distractors: int = 5, p: float = 0.8, sampler: Optional[_SamplingStrategy] = None, **kwargs
    ):
        """Initialize the RaftAnswer generator.

        Args:
            chunks: Dataset containing chunks of text
            n: Number of examples to generate
            distractors: Number of distractor documents
            p: Probability of including oracle document
            sampler: Custom sampling strategy (defaults to random.sample)
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.n = n
        self.distractors = distractors
        self.p = p
        self.chunks = chunks
        self.sampler = sampler or random.sample
        self.formatter = _ContextFormatter()

        self._get_document_set = functools.lru_cache(maxsize=128)(self._get_document_set)

    def prompt(self, input: dict) -> str:
        """Generate prompt for the question."""
        content = self.chunks[input["chunk_id"]]["content"]
        return _DEFAULT_ANSWER_PROMPT.format(question=input["question"], context=content)

    def parse(self, input: dict, response: str) -> Dict[str, str]:
        """Parse response and generate dataset with context."""
        chunk_id = input["chunk_id"]
        oracle_document = self.chunks[chunk_id]["content"]

        # Get document set with sampling logic
        doc_set = self._get_document_set(chunk_id, oracle_document)

        # Format documents as context
        context = self.formatter(doc_set.documents)

        # Create metadata
        title_placeholders = ["placeholder_title"] * len(doc_set.documents)
        metadata = {"title": [title_placeholders], "sentences": [doc_set.documents]}
        instruction = context + "\n" + input["question"]

        return {
            "question": input["question"],
            "cot_answer": response,
            "oracle_document": oracle_document,
            "context": metadata,
            "instruction": instruction,
            "oracle_present": doc_set.oracle_present,
            "oracle_index": doc_set.oracle_index if doc_set.oracle_present else -1,
        }

    def _get_document_set(self, chunk_id: ChunkId, oracle_document: Content) -> DocumentSet:
        """Get a set of documents including the oracle and distractors.

        Returns:
            DocumentSet with documents, oracle index, and whether oracle is present
        """
        chunk_count = len(self.chunks)
        available_indices = [i for i in range(chunk_count) if i != chunk_id]

        # Determine if oracle should be included
        oracle_present = random.random() < self.p

        if oracle_present:
            # Include oracle document with distractors
            documents = [oracle_document]
            distractor_indices = self.sampler(available_indices, self.distractors)

            for idx in distractor_indices:
                documents.append(self.chunks[idx]["content"])

            # Shuffle documents to randomize oracle position
            indices = list(range(len(documents)))
            pairs = list(zip(documents, indices))
            random.shuffle(pairs)
            documents, shuffled_indices = zip(*pairs) if pairs else ([], [])
            documents, shuffled_indices = list(documents), list(shuffled_indices)

            oracle_index = shuffled_indices[0]  # Track where oracle ended up

        else:
            # Select distractors only (oracle + 1 to replace oracle)
            distractor_indices = self.sampler(available_indices, self.distractors + 1)
            documents = [self.chunks[idx]["content"] for idx in distractor_indices]
            random.shuffle(documents)
            oracle_index = -1  # No oracle present

        return DocumentSet(documents=documents, oracle_index=oracle_index, oracle_present=oracle_present)


def chunk_text(text: str, chunk_size: int) -> datasets.Dataset:
    """Splits text into chunks of given size and returns as an HF dataset."""
    chunks = []
    for idx, i in enumerate(range(0, len(text), chunk_size)):
        chunks.append({"chunk_id": idx, "content": text[i : i + chunk_size]})
    return datasets.Dataset.from_list(chunks)


@dataclass
class Raft:
    """RAFT class to generate structured QA datasets from text.

    Args:
        model: Model name to use for question and answer generation
        distractors: Number of distractor documents to include
        chunk_size: Size of text chunks to generate questions for
        n_questions: Number of questions to generate for each chunk
        p: Probability of including oracle document
        backend: Backend to use for question and answer generation
        backend_params: Backend specific parameters
        generation_params: Generation specific parameters
    """

    model: str
    distractors: int = 3
    chunk_size: int = 1000
    n_questions: int = 2
    p: float = 0.8
    backend: str | None = None
    backend_params: dict | None = None
    generation_params: dict | None = None

    def __call__(self, text: str | List[str]) -> datasets.Dataset:
        """Processes text into structured HF dataset."""
        if isinstance(text, str):
            chunks = chunk_text(text, self.chunk_size)
        else:
            chunks = datasets.Dataset.from_list([{"chunk_id": i, "content": t} for i, t in enumerate(text)])

        question_gen = _RaftQuestion(model_name=self.model, backend=self.backend, backend_params=self.backend_params, generation_params=self.generation_params)
        questions = question_gen(chunks)

        answer_gen = _RaftAnswer(
            chunks=chunks, model_name=self.model, backend=self.backend, backend_params=self.backend_params, generation_params=self.generation_params
        )
        qas = answer_gen(questions)
        return qas
