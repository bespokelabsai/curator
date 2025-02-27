"""Hello world example for code execution.

This example demonstrates how to execute a simple Python code snippet on a dataset.
It shows how to define a custom executor class that inherits from curator.CodeExecutor,
and how to implement the code, code_input, and code_output methods.
"""

from datasets import Dataset

from bespokelabs import curator


# Define a custom executor class that inherits from curator.CodeExecutor
class HelloExecutor(curator.CodeExecutor):
    """Executor for the hello world example.

    This class extends curator.CodeExecutor and implements the code, code_input, and code_output methods.
    """

    def code(self, row):
        """Define the Python code to be executed.

        Args:
            row: The dataset row

        Returns:
            The Python code to be executed
        """
        return """location = input();print(f"Hello {location}")"""

    def code_input(self, row):
        """Provide the input to the code execution.

        Args:
            row: The dataset row

        Returns:
            The input to the code execution
        """
        return row["location"]

    def code_output(self, row, execution_output):
        """Process the output from code execution.

        Args:
            row: The dataset row
            execution_output: The output from code execution

        Returns:
            The row with the output added
        """
        row["output"] = execution_output.stdout
        return row


# Create a simple dataset with two locations
locations = Dataset.from_list([{"location": "New York"}, {"location": "San Francisco"}])

# Initialize executor
hello_executor = HelloExecutor()

# Execute the code for each row in the dataset and print the results as a pandas DataFrame
print(hello_executor(locations).to_pandas())
