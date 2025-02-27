# Math Animation Code Execution Example

This example demonstrates how to use Curator's code execution functionality to generate mathematical animations using the Manim library. It shows a complete pipeline for generating, executing, and publishing math animations.

## Overview

The example shows how to:
1. Generate mathematical concepts and explanations using LLMs
2. Convert these concepts into executable Manim code
3. Execute the code in a Docker container with Manim installed
4. Extract the generated videos and publish them to the Hugging Face Hub

## Components

- `script_generator.py`: Generates mathematical concepts and detailed outlines
- `generate_script.py`: Creates hierarchical math content with subjects, topics, and questions
- `generate_manim_code.py`: Converts math concepts into executable Manim code
- `execute_code.py`: Runs the Manim code in a Docker container and extracts videos

## Setup

Install dependencies:

```bash
pip install bespokelabs-curator datasets
```

## Running the Example

```bash
python generate_script.py --num_subjects 20 --topics_per_subject 5 --questions_per_topic 3 --output_dataset_name pimpalgaonkar/math_scripts
python generate_manim_code.py --dataset_name pimpalgaonkar/math_scripts --output_dataset_name pimpalgaonkar/manim_codes
python execute_code.py --dataset_name pimpalgaonkar/manim_codes --output_dataset_name pimpalgaonkar/manim_animations
```

## Output

The example will generate a dataset of math animations and publish them to the Hugging Face Hub.