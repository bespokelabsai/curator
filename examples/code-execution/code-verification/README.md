# Code Verification Example

This example demonstrates how to use Curator's code execution functionality with Ray backend for executing code on large datasets. It shows how to verify code solutions against expected outputs, which is useful for evaluating code generation models or testing programming assignments. This approach is used in projects like [OpenThoughts](https://www.open-thoughts.ai/) and [Bespoke-Stratos](https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation) for code evaluation.

## Overview

The example shows how to:
1. Define a custom code executor for verification tasks
2. Extract code and test inputs from a dataset
3. Execute the code in an isolated environment
4. Compare execution results with expected outputs
5. Process the results at scale using Ray

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Example

```bash
python code_verify.py
```

The output will be a pandas DataFrame with the results of the verification with a `correct` column indicating whether the code solution is correct.
