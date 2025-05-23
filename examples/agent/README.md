# Agent Multi-Turn Example

This example demonstrates a simple conversation between a patient and a doctor using Curator's agent framework.

## Overview

The example implements a conversation between:
- A `Patient` agent that represents a person seeking medical advice
- A `Doctor` agent that provides medical guidance
- Both agents use LLMs to generate responses in a natural conversation flow

## Prerequisites

- Python 3.x
- Curator package installed
- Access to LLM API (OpenAI, Anthropic, etc.)

## Usage

Run the conversation using:

```bash
python multiturn.py
```

This will:
1. Create patient and doctor agents using the specified LLM (default: GPT-4)
2. Start a conversation with the patient describing their symptoms
3. Allow the doctor to ask questions and provide advice
4. Print the complete conversation history

## Configuration

You can modify the following in the code:
- `model_name`: Change the LLM model (default: "gpt-4")
- `max_length`: Maximum number of conversation turns (default: 5)
- `seed_message`: Initial message from the patient
- System prompts for both agents to customize their behavior

## Code Structure

- `Patient` and `Doctor` classes: Agent implementations
- System prompts define the role and behavior of each agent
- `MultiTurnAgents` handles the conversation flow
- Simple console output of the conversation history

## Notes

- Rich display is disabled by default for cleaner output
- The doctor agent is programmed to recommend seeing a real doctor for serious concerns