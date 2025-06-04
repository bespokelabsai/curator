"""Extract text from a prescription PDF."""

import glob

from datasets import Dataset
from pydantic import BaseModel

from bespokelabs import curator


class PrescriptionFormat(BaseModel):
    """A prescription format."""

    name: str
    hospital: str
    patient_name: str
    patient_age: int
    patient_gender: str
    diagnosis: str
    medication: list[str]


class PrescriptionExtractor(curator.LLM):
    """A prescription extractor that extracts text from a prescription PDF."""

    response_format = PrescriptionFormat

    def prompt(self, input: dict) -> str:
        """Generate a prompt using the ingredients."""
        prompt = "Extract text from the prescription PDF."
        return prompt, curator.types.File(url=input["pdf_path"])

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""
        return response


def main():
    """Extract text from a prescription PDF."""
    prescriptions = [
        {"pdf_path": f"./examples/multimodal/prescription-extraction/{path}"} for path in glob.glob("./examples/multimodal/prescription-extraction/data/*")
    ]
    prescriptions = Dataset.from_list(prescriptions)
    prescription_extractor = PrescriptionExtractor(
        model_name="gemini/gemini-2.0-flash",
        backend="litellm",
    )

    prescriptions = prescription_extractor(prescriptions)

    # Print results
    print(prescriptions.dataset.to_pandas())


if __name__ == "__main__":
    main()
