"""Example of using the curator library to generate PII detection dataset.

We use pre-generated Chinese entities to create paragraphs that naturally incorporate
these entities, then track which entities were used in the generation."""

from typing import List, Dict
import json
import os
from pathlib import Path

from datasets import Dataset

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.environ["openai_key_4"]
from pydantic import BaseModel, Field

from bespokelabs import curator


class PIIEntity(BaseModel):
    """A single PII entity that was detected in the text."""
    entity_type: str = Field(description="Type of PII entity (e.g., name, address, phone)")
    entity_value: str = Field(description="The actual PII value found in the text")
    entity_source: str = Field(description="The original language of the entity (chinese/english)")


class GeneratedText(BaseModel):
    """The generated text containing PII entities."""
    text: str = Field(description="The generated paragraph containing PII entities")

    @classmethod
    def from_llm_response(cls, response: str) -> "GeneratedText":
        """Parse the LLM response to extract text between <text> tags."""
        import re
        match = re.search(r'<text>\s*(.*?)\s*</text>', response, re.DOTALL)
        if match:
            return cls(text=match.group(1).strip())
        return cls(text=response.strip())


def load_entities() -> Dataset:
    """Load the pre-generated Chinese entities from JSON file."""
    with open("chinese_entities.json", "r", encoding="utf-8") as f:
        entities = json.load(f)
    return Dataset.from_list([{"entity": entity} for entity in entities])


# Define prompts for different types of content
PROMPT_TEMPLATES = [
    "Generate a professional biography paragraph in Chinese that naturally incorporates the following details:\n{entity_details}\n\nMake sure to use these exact values in the text.",
    "Write a business email in Chinese that mentions the following person and their details:\n{entity_details}\n\nEnsure to maintain formality and use the exact values provided.",
    "Create a legal contract template in Chinese that includes the following person's information:\n{entity_details}\n\nMake sure to use these exact values in appropriate legal context.",
    "Write a news article in Chinese about a business deal, incorporating these details:\n{entity_details}\n\nUse the exact values provided naturally in the text.",
    "Generate a bilingual (English/Chinese) social media post announcing a new hire with these details:\n{entity_details}\n\nEnsure both languages contain the exact values provided."
]


def format_entity_details(entity: Dict) -> str:
    """Format entity details for prompt insertion."""
    details = [
        f"姓名 (Name): {entity['chinese_name']} / {entity['english_name']}",
        f"性别 (Gender): {entity['gender']}",
        f"职业 (Occupation): {entity['occupation']}",
        f"电话 (Phone): {entity['phone']}",
        f"邮箱 (Email): {entity['email']}",
        f"地址 (Address): {entity['address']['chinese']}"
    ]
    return "\n".join(details)


# Create the paragraph generator
paragraph_generator = curator.LLM(
    prompt_func=lambda row: f"""You are a professional writer creating content in Chinese and English. Your task is to generate text that naturally incorporates personal information.

TASK:
{PROMPT_TEMPLATES[row["prompt_index"]].format(entity_details=format_entity_details(row["entity"]))}

REQUIREMENTS:
1. Use ALL the provided entity values exactly as given
2. Write naturally flowing text that incorporates the information
3. Return ONLY the generated text, no explanations or metadata
4. Ensure the text is appropriate for the prompt type (biography, email, contract, etc.)

FORMAT YOUR RESPONSE LIKE THIS:
<text>
[Your generated text here]
</text>
""",
    model_name="gpt-4-1106-preview",  # Using latest GPT-4 for better Chinese language generation
    response_format=None,  # We'll parse the response manually
    parse_func=lambda row, response: {
        "input": GeneratedText.from_llm_response(response).text,
        "output": [
            {
                "entity_type": "name",
                "entity_value": row["entity"]["chinese_name"],
                "entity_source": "chinese"
            },
            {
                "entity_type": "name",
                "entity_value": row["entity"]["english_name"],
                "entity_source": "english"
            },
            {
                "entity_type": "occupation",
                "entity_value": row["entity"]["occupation"],
                "entity_source": "chinese"
            },
            {
                "entity_type": "phone",
                "entity_value": row["entity"]["phone"],
                "entity_source": "mixed"
            },
            {
                "entity_type": "email",
                "entity_value": row["entity"]["email"],
                "entity_source": "mixed"
            },
            {
                "entity_type": "address",
                "entity_value": row["entity"]["address"]["chinese"],
                "entity_source": "chinese"
            }
        ]
    }
)


def main():
    # Load our pre-generated entities
    entities = load_entities()
    
    # Create a dataset with entities and prompt indices
    # For each entity, we'll generate a paragraph using each prompt type
    full_dataset = Dataset.from_list([
        {"entity": entity, "prompt_index": i}
        for entity in entities["entity"]
        for i in range(len(PROMPT_TEMPLATES))
    ])
    
    # Generate paragraphs with PII entities
    pii_dataset = paragraph_generator(full_dataset)
    
    # Save the resulting dataset
    pii_dataset.save_to_disk("pii_dataset")
    print(f"Generated {len(pii_dataset)} examples")
    print("\nExample entry:")
    print(json.dumps(pii_dataset[0], ensure_ascii=False, indent=2))




if __name__ == "__main__":
    main()
