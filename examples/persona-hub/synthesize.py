"""Reimplementation of persona-hub openai_synthesize.py using curator.

This script generates text using different templates and a persona dataset. It supports
multiple template types (instruction, knowledge, npc, math) and uses the curator library
for text generation.

Source: https://github.com/tencent-ailab/persona-hub/blob/main/code/openai_synthesize.py

How to run:
`python synthesize.py --template "math" --output_path "math.jsonl"`
Use `curator-viewer` to view the output.
"""

import argparse

import prompt_templates
from datasets import load_dataset

from bespokelabs import curator


def get_template(template_name):
    """Get the appropriate template based on the template name.

    Args:
        template_name: Name of the template to use ('instruction', 'knowledge', 'npc', or 'math')

    Returns:
        str: The template string to use for generation

    Raises:
        ValueError: If an invalid template type is provided
    """
    # Load the appropriate template
    if template_name == "instruction":
        return prompt_templates.instruction_template
    elif template_name == "knowledge":
        return prompt_templates.knowledge_template
    elif template_name == "npc":
        return prompt_templates.npc_template
    elif template_name == "math":
        return prompt_templates.math_template
    else:
        raise ValueError("Invalid template type. Choose from 'instruction', 'knowledge', 'npc', or 'math'.")


def get_generator(template):
    """Create a text generator using the specified template.

    Args:
        template: Template string to use for generation

    Returns:
        curator.LLM: A configured LLM generator
    """

    class PersonaGenerator(curator.LLM):
        """A text generator that uses a template to generate text based on personas."""

        @classmethod
        def prompt(cls, input: dict) -> str:
            """Generate a prompt using the template and persona."""
            return template.format(persona=input["persona"])

    return PersonaGenerator(
        model_name="gpt-4o",
        generation_params={"temperature": 0.7},
    )


def main(args):
    """Main function to run the text generation pipeline.

    Args:
        args: Parsed command line arguments containing:
            - template: Template type to use
            - sample_size: Number of samples to process
            - output_path: Path to save the generated output
    """
    template = get_template(args.template)
    generator = get_generator(template)
    # Load the persona dataset
    persona_dataset = load_dataset("proj-persona/PersonaHub", data_files="persona.jsonl", split="train")
    if args.sample_size > 0:
        persona_dataset = persona_dataset.take(args.sample_size)
    print(f"Total number of input personas: {len(persona_dataset['persona'])}")
    output = generator(persona_dataset)
    # You can now view this via the curator-viewer (use `curator-viewer` command) or store directly to hf hub.
    # Store the hf dataset to jsonl file.
    output.to_json(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize text using a specified model and template.")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10,
        help=("Number of samples to process from the dataset; Set it to 0 if you want to use the full set of 200k personas."),
    )
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        choices=["instruction", "knowledge", "npc", "math"],
        help=("Prompt templates. Choose from 'instruction', 'knowledge', 'math' or 'npc'. You can also add more customized templates in prompt_templates.py"),
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file.")

    args = parser.parse_args()
    main(args)
