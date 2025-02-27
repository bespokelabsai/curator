# Chart Generation Example

This example demonstrates how to use the Curator Code Executor to generate charts and capture the resulting images as part of a dataset.

## Overview

The example shows how to:
1. Define a custom code executor for chart generation
2. Execute Python code that creates matplotlib charts
3. Capture and process the generated chart images
4. Store the results in a dataset

## How It Works

The `ChartCodeExecutor` class extends Curator's `CodeExecutor` base class and implements three key methods:

- `code()`: Defines the Python code to be executed, which creates a matplotlib chart
- `code_input()`: Prepares the input data for the chart (x/y values, title, labels)
- `code_output()`: Processes the execution results, extracting the generated chart image

## Running the Example

To run this example:

1. Install the required dependencies:

```bash
pip install bespokelabs-curator datasets
```

2. Run the example:

```bash
python charts.py
```

This will execute the chart generation code and save the resulting images to the dataset.