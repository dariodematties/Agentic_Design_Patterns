# Agentic Design Patterns

This repository contains two small LangChain examples built with OpenAI models.

## Files

- `Code_1.py`: a simple two-step LCEL chain that:
- extracts technical specifications from free text
- transforms them into a JSON-like output with `cpu`, `memory`, and `storage`

- `Code_2.py`: a simple coordinator-style example that:
- classifies a user request as `booking`, `info`, or `unclear`
- routes the request to a simulated handler
- prints the delegated result

## Setup

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt

## Environment Variables

Both scripts use OpenAI and require:

export OPENAI_API_KEY="your_api_key_here"

## Run

Run the first example:

python Code_1.py

Run the second example:

python Code_2.py

## Notes

- Code_1.py uses langchain_openai.ChatOpenAI to run a basic sequential LCEL workflow.
- Code_2.py uses langchain_openai.ChatOpenAI plus RunnableBranch to simulate a coordinator delegating work to sub-agents.
- requirements.txt currently still includes langchain-google-genai, although the current version of Code_2.py no longer uses
it.

