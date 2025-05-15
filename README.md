# biomcp-trialgpt

[![Release](https://img.shields.io/github/v/release/genomoncology/biomcp-trialgpt)](https://img.shields.io/github/v/release/genomoncology/biomcp-trialgpt)
[![Build status](https://img.shields.io/github/actions/workflow/status/genomoncology/biomcp-trialgpt/main.yml?branch=main)](https://github.com/genomoncology/biomcp-trialgpt/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/genomoncology/biomcp-trialgpt/branch/main/graph/badge.svg)](https://codecov.io/gh/genomoncology/biomcp-trialgpt)
[![Commit activity](https://img.shields.io/github/commit-activity/m/genomoncology/biomcp-trialgpt)](https://img.shields.io/github/commit-activity/m/genomoncology/biomcp-trialgpt)
[![License](https://img.shields.io/github/license/genomoncology/biomcp-trialgpt)](https://img.shields.io/github/license/genomoncology/biomcp-trialgpt)

Demonstration of TrialGPT agents using BioMCP.

- **Github repository**: <https://github.com/genomoncology/biomcp-trialgpt/>
- **Documentation** <https://genomoncology.github.io/biomcp-trialgpt/>

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:genomoncology/biomcp-trialgpt.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version

- Create an API Token on [PyPI](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting [this page](https://github.com/genomoncology/biomcp-trialgpt/settings/secrets/actions/new).
- Create a [new release](https://github.com/genomoncology/biomcp-trialgpt/releases/new) on Github.
- Create a new tag in the form `*.*.*`.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/cicd/#how-to-trigger-a-release).

## BioMCP TrialGPT

A demonstration of TrialGPT agents using BioMCP for clinical trial matching.

## Getting Started

### Prerequisites

Before running the application, ensure you have:

1. Python 3.9 or higher installed
2. The following API keys:
   - `OPENAI_API_KEY` - Required for all agent frameworks
   - `ANTHROPIC_API_KEY` - Required for Claude models
   - `GEMINI_API_KEY` - Required for Google Gemini models

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/genomoncology/biomcp-trialgpt.git
   cd biomcp-trialgpt
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```

3. Install the package and dependencies:
   ```bash
   pip install -e .
   ```

### Running the Application

1. Set the required API keys as environment variables:
   ```bash
   # On macOS/Linux
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   export GEMINI_API_KEY="your-gemini-api-key"
   
   # On Windows (Command Prompt)
   set OPENAI_API_KEY=your-openai-api-key
   set ANTHROPIC_API_KEY=your-anthropic-api-key
   set GEMINI_API_KEY=your-gemini-api-key
   
   # On Windows (PowerShell)
   $env:OPENAI_API_KEY="your-openai-api-key"
   $env:ANTHROPIC_API_KEY="your-anthropic-api-key"
   $env:GEMINI_API_KEY="your-gemini-api-key"
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run src/biomcp_trialgpt/streamlit_app/main.py
   ```

3. The application will open in your default web browser at `http://localhost:8501`

## Using the Application

1. Enter a clinical note in the text area on the sidebar
2. Select an extraction model (GPT, Claude, or Gemini)
3. Configure trial filtering options (optional)
4. Click "Submit" to process the note and find matching clinical trials
5. View the results in the expandable sections for each step of the process

## Available Agent Frameworks

The application supports multiple agent frameworks:

- **Pydantic Agent**: Uses Pydantic for structured data validation
- **LangGraph Agent**: Uses LangGraph for workflow orchestration

## Troubleshooting

- If you encounter errors about missing API keys, ensure all three keys are properly set as environment variables
- For model-specific errors, check that you have access to the selected models in your API accounts
- If the application fails to start, verify that all dependencies are correctly installed

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
