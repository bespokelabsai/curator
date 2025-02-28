"""Math Animation Code Executor.

This module implements a CodeExecutor for running Manim code to generate math animations.
It executes the code in a Docker container, extracts the generated videos, and adds them
to the dataset for publishing to the Hugging Face Hub.
"""

import argparse
import os
import tarfile
import tempfile
from io import BytesIO
from typing import Dict

from datasets import load_dataset

from bespokelabs import curator


class ManimCodeExecutor(curator.CodeExecutor):
    """CodeExecutor for running Manim code to generate math animations.

    This class extends curator.CodeExecutor and implements the code, code_input, and code_output methods.
    """

    def code(self, input: Dict) -> str:
        """Wrap the manim code in a rendering function.

        Args:
            input: The input dictionary

        Returns:
            The rendering code
        """
        manim_code = input["python_code"]
        rendering_code = f"""
# Function to render the animation
def render_animation():
    import os
    import subprocess
    # Save the code to a temporary file
    with open('temp_animation.py', 'w') as f:
        f.write('''
{manim_code}
''')
    # Run manim to render the animation
    cmd = f"manim temp_animation.py {input['scene_class_name']} -qm -o video.mp4"
    print(f"Running: {{cmd}}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

# Call the function to render
if __name__ == "__main__":
    render_animation()
"""
        return rendering_code

    def code_output(self, input: Dict, output) -> Dict:
        """Postprocess the code execution output.

        Args:
            input: The input dictionary
            output: The output from code execution

        Returns:
            The input dictionary with the video added
        """
        if output.files:
            # Create a temporary directory to work with files
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Handle the bytes string
                    import ast

                    bytes_io = BytesIO(ast.literal_eval(output.files))
                    with tarfile.open(fileobj=bytes_io) as tar:
                        tar.extractall(path=temp_dir)

                    # get the video file
                    video_file = os.path.join(temp_dir, "workspace/media/videos/temp_animation/720p30/video.mp4")
                    if os.path.exists(video_file):
                        # Copy the video file to the output dataset
                        input["video"] = open(video_file, "rb").read()
                    else:
                        # raise Exception("Video file not found")
                        input["video"] = None

            except Exception:
                import traceback

                traceback.print_exc()
                print("Error extracting files")
        # If no files were found, return the input with an error message

        return input


def execute_manim_code(dataset_name="pimpalgaonkar/manim_codes_10k", output_dataset_name="pimpalgaonkar/manim_animations_10k"):
    """Execute manim code for all items in the dataset and save the resulting videos.

    Args:
        dataset_name: Name of the dataset containing manim code
        output_dataset_name: Name for the output dataset on the Hub

    Returns:
        The processed dataset with video paths
    """
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Initialize the code executor
    executor = ManimCodeExecutor(
        backend="docker",
        backend_params={
            "docker_image": "manimcommunity/manim:latest",
        },
    )

    # Create a temporary directory for output files
    os.makedirs("temp_output", exist_ok=True)

    # Process the dataset
    print(f"Executing manim code for {len(dataset['train'])} items...")
    results = executor(
        dataset["train"],
        execution_params={
            "timeout": 120,
        },
    )

    # Push the results to the Hub
    results.push_to_hub(output_dataset_name)
    print(f"Results pushed to {output_dataset_name}")

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Execute Manim code from a dataset and save the resulting videos.")
    parser.add_argument("--dataset_name", type=str, default="pimpalgaonkar/math_codes_dataset", help="Path or name of the dataset containing manim code")
    parser.add_argument("--output_dataset_name", type=str, default="pimpalgaonkar/math_animations_dataset", help="Name for the output dataset on the Hub")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Execute the manim code on our dataset with provided arguments
    execute_manim_code(dataset_name=args.dataset_name, output_dataset_name=args.output_dataset_name)
