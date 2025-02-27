# Hello World Example

This example demonstrates the most basic usage of Curator's code execution functionality. It shows how to create a simple code executor that runs Python code to generate greetings for different locations.

## Overview

The example shows how to:
1. Define a custom code executor by extending the `CodeExecutor` base class
2. Implement the required methods: `code()`, `code_input()`, and `code_output()`
3. Execute the code on a simple dataset
4. Process and display the results

## How It Works

The `HelloExecutor` class:
- Takes location names from a dataset
- Executes Python code that reads the location and prints a greeting
- Captures the output and adds it to the dataset

## Running the Example

To run this example:

1. Install the required dependencies:

```bash
pip install bespokelabs-curator datasets
```

2. Run the example:

```bash
python main.py
```

### Example Output

The output will be a pandas DataFrame with the location and the generated greeting:

```
location	output
0	New York	Hello New York
1	San Francisco	Hello San Francisco
```

This demonstrates the basic workflow for executing code on a dataset using Curator's code execution functionality.