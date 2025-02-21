import json
import os
import tarfile
import tempfile
from io import BytesIO

from datasets import Dataset, Image

from bespokelabs import curator


class ChartCodeExecutor(curator.CodeExecutor):
    """Chart Generation Code Executor."""

    def code(self, row):
        """Extract code string from a dataset row."""
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
        """Prepare chart data as input."""
        chart_data = {"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10], "title": "Sample Line Chart", "xlabel": "X Axis", "ylabel": "Y Axis"}
        return json.dumps(chart_data)

    def code_output(self, row, execution_output):
        """Store execution results."""
        row["chart_generated"] = execution_output.files is not None

        # Create a file-like object from bytes
        with tempfile.TemporaryDirectory() as temp_dir:
            # Handle the bytes string
            if isinstance(execution_output.files, bytes):
                try:
                    # Try to decode and clean up the string
                    files_str = execution_output.files.decode("utf-8")
                    # Remove any b'' wrapping and escape characters
                    files_str = files_str.strip("b'").strip("'").encode("latin1")
                    tar_bytes = BytesIO(files_str)
                except Exception:
                    # Fallback to direct bytes if decoding fails
                    tar_bytes = BytesIO(execution_output.files)
            else:
                tar_bytes = BytesIO(execution_output.files)

            with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
                tar.extractall(path=temp_dir)
            with open(os.path.join(temp_dir, "chart.png"), "rb") as f:
                row["chart_image"] = f.read()

        return row


if __name__ == "__main__":
    # Initialize executor with any backend
    executor = ChartCodeExecutor(backend="multiprocessing")
    # executor = ChartCodeExecutor(backend='ray')
    # executor = ChartCodeExecutor(backend="docker", backend_params={"docker_image": "andgineer/matplotlib"})

    # e2b currently doesn't support files
    # executor = ChartCodeExecutor(backend="e2b")

    # Create sample dataset
    dataset = Dataset.from_list([{"id": 1}])

    # Execute chart generation
    results = executor(dataset)

    # Cast the chart_image column to Image type
    results = results.cast_column("chart_image", Image())
