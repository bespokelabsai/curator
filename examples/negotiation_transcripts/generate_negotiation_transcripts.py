"""Script to generate negotiation transcripts using Curator.

This script generates pairs of negotiation transcripts and their analyses,
where each transcript subtly violates one or two negotiation principles while
maintaining natural dialogue flow.
"""

from typing import List, Dict
from pydantic import BaseModel, Field
from datasets import Dataset

from bespokelabs import curator


class NegotiationTranscript(BaseModel):
    """Model for a negotiation transcript between a coach and client."""
    transcript_text: str = Field(
        description="The full text of the negotiation transcript."
    )
    violated_principle: str = Field(
        description="The principle being subtly violated in this transcript."
    )


class NegotiationAnalysis(BaseModel):
    """Model for analyzing a negotiation transcript."""
    analysis_text: str = Field(
        description="Detailed analysis of how the transcript violates the specified "
        "principle."
    )


# List of principles that can be violated
NEGOTIATION_PRINCIPLES = [
    "Delay Immediate Advice-Giving",
    "Gather Comprehensive Information",
    "Frame Advice as Questions",
    "Encourage Reflective Dialogue",
    "Avoid Assumptions",
    "Maintain Credibility",
    "Promise of Finality Builds Trust",
    "Leverage Closure for Concessions",
    "Timing of Closure is Key",
    "Universal Application"
]

# Prompt template for generating transcripts
TRANSCRIPT_PROMPT_TEMPLATE = """Generate a natural dialogue between a negotiation coach ("Coach") and a client ("Client") that subtly violates the principle: "{principle}".

The dialogue should:
1. Maintain a natural conversation flow
2. Only ask one question at a time
3. Show the coach's expertise while subtly violating the principle
4. Keep other negotiation principles intact
5. Use "Start of Transcript" and "End of Transcript" markers

The violation should be subtle - the coach should still appear professional and competent.

Principle to violate: {principle}
"""

# Prompt template for generating analysis
ANALYSIS_PROMPT_TEMPLATE = """Analyze the following negotiation transcript, focusing on how it violates the principle: "{principle}".

Transcript:
{transcript}

Provide a detailed analysis that includes:
1. The primary principle violated
2. Specific examples of violations
3. Why the violation is subtle
4. Principles successfully followed
5. Impact of the violation
6. Learning points

Format the analysis with clear sections and bullet points.
"""

# LLM for transcript generation
transcript_generator = curator.LLM(
    prompt_func=lambda row: TRANSCRIPT_PROMPT_TEMPLATE.format(
        principle=row["principle"]
    ),
    model_name="gpt-4",
    response_format=NegotiationTranscript,
    parse_func=lambda row, response: {
        "transcript_text": str(response),  # Convert full response to string
        "violated_principle": row["principle"]
    }
)

# LLM for analysis generation
analysis_generator = curator.LLM(
    prompt_func=lambda row: ANALYSIS_PROMPT_TEMPLATE.format(
        principle=row["violated_principle"],
        transcript=row["transcript_text"]
    ),
    model_name="gpt-4",
    response_format=NegotiationAnalysis,
    parse_func=lambda row, response: {
        "analysis_text": str(response)  # Convert full response to string
    }
)


def generate_transcript_and_analysis(principle: str) -> tuple[str, str]:
    """Generate a transcript and its analysis for a given principle."""
    # Create input dataset with the principle
    input_data = Dataset.from_dict({"principle": [principle]})

    # Generate transcript
    transcript_dataset = transcript_generator(input_data)
    transcript = transcript_dataset[0]["transcript_text"]

    # Generate analysis
    analysis_dataset = analysis_generator(transcript_dataset)
    analysis = analysis_dataset[0]["analysis_text"]

    return transcript, analysis


def main():
    """Generate 10 transcript-analysis pairs."""
    import os

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate transcripts for each principle
    for i, principle in enumerate(NEGOTIATION_PRINCIPLES[:10], 1):
        transcript, analysis = generate_transcript_and_analysis(principle)
        
        # Save transcript
        transcript_path = os.path.join(output_dir, f"transcript{i}.txt")
        with open(transcript_path, "w") as f:
            f.write(transcript)
            
        # Save analysis
        analysis_path = os.path.join(output_dir, f"transcript{i}-graded.txt")
        with open(analysis_path, "w") as f:
            f.write(analysis)
        
        print(f"Generated transcript {i} violating principle: {principle}")


if __name__ == "__main__":
    main()
