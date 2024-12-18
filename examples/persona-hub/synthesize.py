""" Reimplementation of persona-hub openai_synthesize.py using curator.
Source: https://github.com/tencent-ailab/persona-hub/blob/main/code/openai_synthesize.py
How to run:
`python synthesize.py --template "math" --output_path "math.jsonl"`
Use `curator-viewer` to view the output.
"""

import argparse
from bespokelabs import curator
from datasets import load_dataset
import prompt_templates


def get_template(template_name):
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
        raise ValueError(
            "Invalid template type. Choose from 'instruction', 'knowledge', 'npc', or 'math'."
        )


def get_generator(template):
    def prompt_func(row):
        return template.format(persona=row["persona"])

    generator = curator.LLM(
        prompt_func=prompt_func,
        model_name="gpt-4o",
        temperature=0.7,
    )
    return generator


def main(args):
    template = get_template(args.template)
    generator = get_generator(template)
    # Load the persona dataset
    persona_dataset = load_dataset(
        "proj-persona/PersonaHub", data_files="persona.jsonl", split="train"
    )
    if args.sample_size > 0:
        persona_dataset = persona_dataset.take(args.sample_size)
    print(f"Total number of input personas: {len(persona_dataset['persona'])}")
    output = generator(persona_dataset)
    # You can now view this via the curator-viewer (use `curator-viewer` command) or store directly to hf hub.
    # Store the hf dataset to jsonl file.
    output.to_json(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize text using a specified model and template."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10,
        help="Number of samples to process from the dataset; Set it to 0 if you want to use the full set of 200k personas.",
    )
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        choices=["instruction", "knowledge", "npc", "math"],
        help=(
            "Prompt templates. Choose from 'instruction', 'knowledge', 'math' or 'npc'. "
            "You can also add more customized templates in prompt_templates.py"
        ),
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file.")

    args = parser.parse_args()
    main(args)
