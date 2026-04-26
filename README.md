# Agentic Design Patterns

This repository contains a small set of Python examples that explore agentic design patterns with:

- LangChain + OpenAI
- Google ADK + Gemini

The code is organized as a progression from simple sequential chains to coordinator, parallel, and reflection-based agent workflows.

## Repository Contents

### `Code_1.py`

A simple sequential LangChain LCEL example using OpenAI.

What it does:
- extracts technical specifications from free text
- transforms the extracted information into a JSON-like structure with `cpu`, `memory`, and `storage`

Key libraries:
- `langchain-openai`
- `langchain-core`

API key:
- `OPENAI_API_KEY`

Run:

```bash
python Code_1.py
```

### `Code_2.py`

A LangChain coordinator-style routing example using OpenAI.

What it does:
- classifies a request into booking or information handling
- routes the request through a coordinator chain
- delegates to simulated Python handlers
- prints the final delegated result

Key libraries:
- `langchain-openai`
- `langchain-core`

API key:
- `OPENAI_API_KEY`

Run:

```bash
python Code_2.py
```

### `Code_3.py`

A Google ADK coordinator example using Gemini.

What it does:
- defines a coordinator agent plus specialized sub-agents
- wraps Python functions as ADK tools
- creates an in-memory session and runner
- delegates booking or information requests through ADK
- prints the final response text

Key libraries:
- `google-adk`
- `google-genai`

API key:
- `GOOGLE_API_KEY` or `GEMINI_API_KEY`

Run:

```bash
python Code_3.py
```

Notes:
- this example depends on Gemini quota and billing availability
- if you see `429 RESOURCE_EXHAUSTED`, the issue is usually project quota rather than Python code
- you may see warnings about `function_call` parts in the returned content; those indicate tool usage, not necessarily failure

### `Code_4.py`

A LangChain parallel-processing example using OpenAI.

What it does:
- runs three chains in parallel for the same topic
- produces:
  - a summary
  - a list of interesting questions
  - a set of key terms
- synthesizes the parallel outputs into a single final response

Key libraries:
- `langchain-openai`
- `langchain-core`

API key:
- `OPENAI_API_KEY`

Run:

```bash
python Code_4.py
```

### `Code_5.py`

A Google ADK parallel-agent example using Gemini and Google Search.

What it does:
- creates three research agents that run in parallel
- each sub-agent researches one sustainability topic:
  - renewable energy
  - electric vehicles
  - carbon capture
- stores each result in shared session state
- runs a synthesis agent after the parallel stage
- prints the final structured report

Key libraries:
- `google-adk`
- `google-genai`

API key:
- `GOOGLE_API_KEY` or `GEMINI_API_KEY`

Run:

```bash
python Code_5.py
```

Notes:
- this example uses the Google Search tool through ADK
- because it is a multi-step parallel workflow, runtime depends on model/tool availability and Gemini quota

### `Code_6.py`

A LangChain reflection-loop example using OpenAI.

What it does:
- asks the model to write a Python `calculate_factorial` function
- runs an iterative generate-and-critique loop
- uses a reviewer prompt to inspect the current code against the original requirements
- stops early if the reviewer returns `CODE_IS_PERFECT`
- prints the refined code after the loop completes

Key libraries:
- `langchain-openai`
- `langchain-core`
- `python-dotenv`

API key:
- `OPENAI_API_KEY`

Run:

```bash
python Code_6.py
```

Notes:
- this example loads environment variables from a local `.env` file if present
- it demonstrates a reflection pattern rather than tool-based multi-agent orchestration

## Setup

Create and activate a virtual environment, then install the base dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Additional Dependency Note

`Code_6.py` imports `python-dotenv` to load `OPENAI_API_KEY` from a local `.env` file. If that package is not already installed in your environment, install it with:

```bash
python -m pip install python-dotenv
```

## Environment Variables

### OpenAI examples

For `Code_1.py`, `Code_2.py`, `Code_4.py`, and `Code_6.py`:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### Gemini / Google ADK examples

For `Code_3.py` and `Code_5.py`:

```bash
export GOOGLE_API_KEY="your_google_api_key"
```

You can also use:

```bash
export GEMINI_API_KEY="your_google_api_key"
```

## Current Dependency Status

The current `requirements.txt` in this repository contains:

```txt
langchain-core
langchain-openai
langchain-google-genai
google-adk
google-genai
```

This means:
- the LangChain and ADK examples are represented in `requirements.txt`
- `langchain-google-genai` is currently present even though the active examples shown in this repository do not depend on it directly
- `python-dotenv` is used by `Code_6.py` but is not currently listed in `requirements.txt`

## Summary

This repository currently demonstrates:

- sequential LangChain workflows
- manual routing with LangChain
- parallel LangChain processing
- reflection loops with LangChain
- coordinator-style agent orchestration with Google ADK
- parallel multi-agent research and synthesis with Google ADK
