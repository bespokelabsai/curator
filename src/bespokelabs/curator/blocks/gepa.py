"""GEPA integration for Curator - enables prompt optimization using evolutionary search.

This module provides a CuratorAdapter that implements GEPA's GEPAAdapter interface,
allowing GEPA to optimize system prompts and prompt templates in Curator LLM classes.

Reference: https://github.com/gepa-ai/gepa
"""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, TypedDict

from datasets import Dataset
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from bespokelabs.curator import LLM


class EvaluationResult(TypedDict):
    """Result from evaluating a single example."""

    score: float
    feedback: str


class CuratorTrajectory(TypedDict):
    """Trajectory data captured during evaluation."""

    input: Dict[str, Any]
    system_prompt: str
    output: Dict[str, Any] | None
    score: float
    feedback: str


class CuratorRolloutOutput(TypedDict):
    """Output from a single LLM rollout."""

    output: Dict[str, Any] | None


@dataclass
class CuratorAdapter(GEPAAdapter[Dict[str, Any], CuratorTrajectory, CuratorRolloutOutput]):
    """GEPA adapter for optimizing Curator LLM prompts.

    This adapter wraps a Curator LLM class and allows GEPA to optimize
    its system prompts and prompt templates through evolutionary search.

    Attributes:
        llm: The Curator LLM instance to optimize (must be an instance of curator.LLM)
        metric: A callable that evaluates outputs. Signature: (inputs, outputs) -> List[EvaluationResult]
                Each EvaluationResult contains 'score' (float) and 'feedback' (str).
                The feedback is used by GEPA's reflection mechanism to propose targeted improvements.

    The adapter automatically extracts from the LLM instance:
        - system_prompt: From the instance's `prompt_formatter.system_prompt` attribute
        - response_format: From the instance's `response_format` attribute (optional)
    """

    llm: LLM
    metric: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], List[EvaluationResult]]

    def __post_init__(self):
        """Validate the adapter configuration."""
        if not isinstance(self.llm, LLM):
            raise TypeError(f"llm must be an instance of curator.LLM, got {type(self.llm)}")

        if not callable(self.metric):
            raise TypeError("metric must be callable")

    def get_seed_candidate(self) -> Dict[str, str]:
        """Extract seed candidate prompts from the LLM class."""
        candidate: Dict[str, str] = {}

        # Extract system_prompt from instance
        # Access via prompt_formatter which is the source of truth
        system_prompt = getattr(self.llm.prompt_formatter, "system_prompt", None)
        if not system_prompt:
            raise ValueError("No system prompt found to optimize")
        candidate["system_prompt"] = system_prompt

        return candidate

    def _create_optimized_llm(self, candidate: Dict[str, str]) -> LLM:
        """Create an LLM instance with the candidate's prompts injected.

        Args:
            candidate: Dict containing prompt components to inject:
                - "system_prompt": Optional system prompt override

        Returns:
            An LLM instance configured with the candidate's prompts
        """
        # Create a clone with the optimized system prompt
        system_prompt = candidate.get("system_prompt")
        new_llm = self.llm.clone(system_prompt=system_prompt)

        return new_llm

    def evaluate(
        self,
        batch: List[Dict[str, Any]],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[CuratorTrajectory, CuratorRolloutOutput]:
        """Evaluate a candidate's prompts on a batch of examples.

        This method:
        1. Creates an LLM instance with the candidate's prompts injected
        2. Runs the LLM on the batch
        3. Scores each output using the metric function
        4. Returns an EvaluationBatch with scores and optional trajectories

        Args:
            batch: List of input examples to evaluate
            candidate: Dict containing prompt components:
                - "system_prompt": Optional system prompt
            capture_traces: If True, populate trajectories for reflection

        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories
        """
        outputs: List[CuratorRolloutOutput] = []
        scores: List[float] = []
        trajectories: List[CuratorTrajectory] | None = [] if capture_traces else None

        # Create the optimized LLM
        llm = self._create_optimized_llm(candidate)

        # Convert batch to Dataset for Curator
        dataset = Dataset.from_list(batch)

        # GEPA requires fresh rollouts per candidate. Curator's cache fingerprint
        # does not include system prompt, so we disable cache for optimization.
        previous_disable_cache = os.environ.get("CURATOR_DISABLE_CACHE")
        os.environ["CURATOR_DISABLE_CACHE"] = "true"
        try:
            result = llm(dataset)
        finally:
            if previous_disable_cache is None:
                os.environ.pop("CURATOR_DISABLE_CACHE", None)
            else:
                os.environ["CURATOR_DISABLE_CACHE"] = previous_disable_cache

        output_dataset = result.dataset

        # Convert outputs to list of dicts
        output_dicts = [dict(row) for row in output_dataset]

        # Evaluate all outputs using the metric function
        eval_results = self.metric(batch, output_dicts)
        if not isinstance(eval_results, list):
            raise TypeError(f"metric function must return list[EvaluationResult], got {type(eval_results)}")
        if len(eval_results) != len(batch):
            raise ValueError(f"metric function must return a list of the same length as the input batch (expected {len(batch)}, got {len(eval_results)})")
        scores = [float(r["score"]) for r in eval_results]

        # Build outputs and trajectories
        for input_row, output_dict, eval_result in zip(batch, output_dicts, eval_results):
            outputs.append({"output": output_dict})

            if capture_traces and trajectories is not None:
                trajectories.append(
                    {
                        "input": input_row,
                        "system_prompt": candidate.get("system_prompt", ""),
                        "output": output_dict,
                        "score": float(eval_result["score"]),
                        "feedback": eval_result["feedback"],
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch[CuratorTrajectory, CuratorRolloutOutput],
        components_to_update: List[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build a reflective dataset for GEPA's instruction proposal.

        This method formats execution traces into a structured dataset that
        GEPA's reflection LLM uses to understand what went wrong and propose
        improvements for each component.

        Args:
            candidate: The candidate that was evaluated
            eval_batch: Result of evaluate(..., capture_traces=True)
            components_to_update: Component names to generate datasets for

        Returns:
            Dict mapping component name to list of reflective records
        """
        result: Dict[str, List[Dict[str, Any]]] = {}

        if eval_batch.trajectories is None:
            raise ValueError("eval_batch must have trajectories (call evaluate with capture_traces=True)")

        for component in components_to_update:
            items: List[Dict[str, Any]] = []

            for traj in eval_batch.trajectories:
                # Build inputs dict based on what's relevant for this component
                inputs: Dict[str, Any] = {"input": traj["input"]}

                if component == "system_prompt":
                    inputs["current_system_prompt"] = traj["system_prompt"]

                items.append(
                    {
                        "Inputs": inputs,
                        "Generated Outputs": traj["output"],
                        "Feedback": traj["feedback"],
                    }
                )

            result[component] = items

        if not any(result.values()):
            raise ValueError("No valid trajectories found for reflection")

        return result
