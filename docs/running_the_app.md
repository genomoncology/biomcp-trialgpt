# Running the BioMCP TrialGPT Application

This guide provides instructions for setting up and running the BioMCP TrialGPT Streamlit application.

## Prerequisites

Before running the application, ensure you have:

1. Python 3.9 or higher installed
2. The following API keys:
   - `OPENAI_API_KEY` - Required for all agent frameworks
   - `ANTHROPIC_API_KEY` - Required for Claude models
   - `GEMINI_API_KEY` - Required for Google Gemini models

## Installation

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

## Running the Application

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
3. Configure trial filtering options (optional):
   - Recruiting Status (OPEN, CLOSED, ANY)
   - Date filters (Min/Max date)
   - Phase (Phase 1, Phase 2, Phase 3, Phase 4, N/A)
4. Select an agent framework (Pydantic or LangGraph)
5. Click "Submit" to process the note and find matching clinical trials
6. View the results in the expandable sections for each step of the process:
   - Step 1: Clinical Note Extraction
   - Step 2: Trial Retrieval
   - Step 3: Eligibility Matching
   - Step 4: Trial Scoring

## Available Agent Frameworks

The application supports multiple agent frameworks:

- **Pydantic Agent**: Uses Pydantic for structured data validation
- **LangGraph Agent**: Uses LangGraph for workflow orchestration

## Troubleshooting

- If you encounter errors about missing API keys, ensure all three keys are properly set as environment variables
- For model-specific errors, check that you have access to the selected models in your API accounts
- If the application fails to start, verify that all dependencies are correctly installed
- Use the "Reset Workflow" button in the sidebar to clear the current session and start over
