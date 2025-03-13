# This example demonstrates how to use a code executor to execute a code that generates a chart
# and capture the chart images as part of a dataset

import ast
import json
import os
import tarfile
import tempfile
from io import BytesIO

from datasets import Dataset, Image

from bespokelabs import curator


class ChartCodeExecutor(curator.CodeExecutor):
    """Chart Generation Code Executor.

    This class executes Python code that generates matplotlib charts and captures
    the resulting images. It demonstrates how to:
    1. Define the code to be executed
    2. Prepare input data for the code
    3. Process the execution results, including capturing generated files
    """

    def code(self, row):
        """Extract code string from a dataset row.

        This method returns the Python code that will be executed.
        The code will:
        - Import necessary libraries (matplotlib, json)
        - Read input data from stdin
        - Create a line chart based on the input data
        - Save the chart as an image file

        Args:
            row: A dataset row (not used in this implementation)

        Returns:
            str: The Python code to execute
        """
        return """
import matplotlib.pyplot as plt
import json

# Read input data
data = json.loads(input())

# Create the chart
plt.figure(figsize=(10, 6))
plt.plot(data['x'], data['y'])
plt.title(data['title'])
plt.xlabel(data['xlabel'])
plt.ylabel(data['ylabel'])

# Save the plot
plt.savefig('chart.png')
plt.close()

"""

    def code_input(self, row):
        """Prepare chart data as input.

        This method creates sample data for the chart and serializes it to JSON.
        In a real application, this could use data from the dataset row.

        Args:
            row: A dataset row (not used in this implementation)

        Returns:
            str: JSON-serialized chart data
        """
        # create sample data for the chart
        chart_data = {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "title": "Sample Line Chart",
            "xlabel": "X Axis",
            "ylabel": "Y Axis",
        }
        return json.dumps(chart_data)

    def code_output(self, row, execution_output):
        """Store execution results.

        This method processes the output from code execution:
        - Checks if files were generated
        - Extracts the chart image from the execution output
        - Stores the chart image in the dataset row

        Args:
            row: The dataset row to update with results
            execution_output: The output from code execution, including any generated files

        Returns:
            dict: The updated dataset row with execution results
        """
        row["chart_generated"] = execution_output.files is not None

        # get the chart image from execution output
        with tempfile.TemporaryDirectory() as temp_dir:
            if execution_output.files:
                # extract tar
                files_str = ast.literal_eval(execution_output.files)
                tar_bytes = BytesIO(files_str)
                with tarfile.open(fileobj=tar_bytes, mode="r") as tar:
                    tar.extractall(path=temp_dir)

                # read the chart image
                # it is in workspace/chart.png for 'docker' backend and chart.png for 'multiprocessing' backend
                try:
                    with open(os.path.join(temp_dir, "workspace/chart.png"), "rb") as f:
                        row["chart_image"] = f.read()
                except Exception:
                    try:
                        with open(os.path.join(temp_dir, "chart.png"), "rb") as f:
                            row["chart_image"] = f.read()
                    except Exception:
                        row["chart_image"] = None

        return row


if __name__ == "__main__":
    # This section demonstrates how to use the ChartCodeExecutor

    # Initialize executor with the multiprocessing backend (runs code in a separate process)
    executor = ChartCodeExecutor(backend="multiprocessing")  # or ray, docker (with matplotlib image)

    # Create a simple dataset with one example
    dataset = Dataset.from_list([{"id": 1}])

    # Execute chart generation for each row in the dataset
    results = executor(dataset)

    # Convert the binary chart_image data to Hugging Face's Image type for better handling
    results = results.cast_column("chart_image", Image())

    # Print the results - in a real application, you might save this dataset or push to Hugging Face Hub
    print(results)
