# ClaimCLAIRE

An enhanced fact-checking system that verifies claims using web search and LLM-based reasoning.

🌐 **Try it online**: [https://claimclaire.vercel.app/](https://claimclaire.vercel.app/)

---

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Acknowledgments](#acknowledgments)
5. [License](#license)

---

## Overview
ClaimCLAIRE is an AI-powered fact-checking system that verifies claims by:

- Decomposing claims into atomic components for systematic verification
- Gathering evidence from web searches with source trust ratings
- Evaluating each component against collected evidence
- Synthesizing a final verdict with detailed explanations and citations

The agent follows a five-stage pipeline:

1. **Claim Decomposition**: Breaks down the input claim into atomic components using iterative validation
2. **Holistic Evidence Gathering**: A ReAct-style agent searches the web using Serper.dev API with LLM-based reranking
3. **Component Evaluation & Gap-Filling**: Evaluates each component against gathered evidence, performs targeted searches for unverified components
4. **Verdict Synthesis**: Applies deterministic logic rules to determine if the claim is consistent or inconsistent
5. **Report Generation**: Generates a natural-language explanation with citations and trust ratings

All web searches are performed using the Serper.dev Google Search API with optional Gemini-based reranking for improved relevance. Sources are assigned trust ratings (Reliable, Mixed, or Unreliable) to weight their credibility.

## Dataset Setup

### Quick Test with Sample Data

A sample dataset (`sample_dev.json`) with 5 examples is included in `data/` for quick testing:

```bash
pixi shell
python evaluate_ablations.py \
  --data-path data/sample_dev.json \
  --ablation A4 \
  --output-path results/sample_test.json
```

### Full Dataset

For complete evaluations, download the full AVeriTeC dataset:

1. Download `dev.json` from [HuggingFace](https://huggingface.co/chenxwh/AVeriTeC/tree/main/data)
2. Place it in the `data/` directory:
   ```bash
   # The data/ directory already exists
   # Download and place dev.json there
   ```

The dataset contains fact-checking claims with labels: "Supported", "Refuted", "Not Enough Evidence", or "Conflicting Evidence/Cherrypicking".

---

## Installation
### 1. Prerequisites
- Linux (tested on Ubuntu 22.04) or macOS.
- Python 3.12 (managed automatically via [pixi](https://pixi.sh)).
- Access credentials for your preferred LLM provider.

### 2. Install pixi (one-time)
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 3. Clone the repository
```bash
git clone https://github.com/yo-lxmmm/ClaimCLAIRE.git
cd ClaimCLAIRE
```

### 4. Create the project environment

**Option A: Using pixi (recommended)**
```bash
pixi shell
```
The first run resolves Python 3.12 and all dependencies defined in `pixi.toml`. Subsequent invocations reuse the cached env.

**Option B: Using pip**
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Configure API Keys
The agent requires API keys for LLM providers and web search. Create a `.env` file in the project root with your credentials. You can use `.env.template` as a starting point:

```bash
cp .env.template .env
```

Then edit `.env` with your credentials.

**Required API Keys:**
```bash
# Serper.dev API key (REQUIRED for web search)
SERPER_API_KEY=your_serper_api_key_here

# Google Gemini API key (if using Google models)
GOOGLE_API_KEY=your_google_api_key_here
```

**Optional API Keys (for other LLM providers):**
```bash
# Anthropic (for Claude models - used in decomposition by default)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-06-01

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here
```

> ℹ️ **Note**: The system uses Claude Sonnet 4 for claim decomposition (A2+) and the specified `--engine` model for other components. Refer to the [LangChain chat model docs](https://python.langchain.com/docs/integrations/chat) for provider-specific variables.


## Quick Start

### Option 0: Try it LIVE!! (No Setup Required)
**Use the live web interface**: [https://claimclaire.vercel.app/](https://claimclaire.vercel.app/)

No installation or API keys needed - just visit the link and start verifying claims!

<p align="center">
  <img src="images/ClaimCLAIRE demo.png" alt="ClaimCLAIRE live demo" width="70%">
</p>

### Option 1: Web Interface (Local)
Run the Flask web application locally for an interactive interface:

```bash
# Inside `pixi shell` or activated venv
python baseline_web_app.py
```

Then open your browser to `http://localhost:8080` to use the web interface.

> ℹ️ **Note**: Make sure you have set `GOOGLE_API_KEY` and `SERPER_API_KEY` in your `.env` file before running the web app.

### Option 2: Python Library Integration
You can integrate the agent directly into your application:

Example code:
```python
from claire_agent import InconsistencyAgent
from utils.report_rendering import render_inconsistency_report
import asyncio

agent = InconsistencyAgent(
    engine="gemini-2.5-flash",
    model_provider="google_genai",
    num_results_per_query=10,
    reasoning_effort=None,
)

async def main():
    claim = "Bernie Sanders purchased an opulent Vermont mansion in 2016 for $2.5 million."

    report = await agent.analyze_claim(
        claim_text=claim,
        passage=claim  # Can provide additional context if available
    )

    render_inconsistency_report(report)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Running Evaluations

### Ablation Studies

The system includes ablation studies (A0-A4) to evaluate different components:

| Ablation | ReAct Agent | Iterative Decomposition | Trust Weighting | Gap-Filling | Description |
|----------|-------------|------------------------|-----------------|-------------|-------------|
| **A0** | ❌ | ❌ | ❌ | ❌ | Baseline RAG (simple search) |
| **A1** | ✅ | ❌ | ❌ | ❌ | A0 + ReAct Agent |
| **A2** | ✅ | ✅ | ❌ | ❌ | A1 + Iterative Decomposition |
| **A3** | ✅ | ✅ | ✅ | ❌ | A2 + Trust Weighting |
| **A4** | ✅ | ✅ | ✅ | ✅ | A3 + Gap-Filling (Full System) |

### Run Single Ablation

Evaluate a specific ablation variant:

```bash
# Inside pixi shell
python evaluate_ablations.py \
  --data-path data/dev.json \
  --ablation A4 \
  --output-path results/ablation_A4_full.json \
  --engine gemini-2.5-flash \
  --model-provider google_genai \
  --num-results 10
```

To test on a subset:
```bash
python evaluate_ablations.py \
  --data-path data/dev.json \
  --ablation A2 \
  --output-path results/ablation_A2_test.json \
  --max-examples 10 \
  --engine gemini-2.5-flash \
  --model-provider google_genai \
  --num-results 10
```

### Run All Ablations

Use the provided script to run all ablations sequentially:

```bash
# Inside pixi shell
./run_evaluations.sh
```

This will run A0, A1, A2, A3, and A4 on the full dataset and save results to the `results/` directory.

### Understanding Results

Each evaluation produces three files:

1. **JSON file** (`ablation_A4_full.json`): Complete results with metrics and predictions
   - `metrics`: Overall accuracy, macro F1, per-class precision/recall/F1
   - `label_breakdown`: Performance broken down by original label
   - `results`: Per-example predictions

2. **Results CSV** (`ablation_A4_full_results.csv`): Main evaluation results
   - Columns: `claim_id`, `claim`, `gold_label`, `predicted_verdict`, `correct`, `num_components`, `num_sources`, `gap_fill_triggered`, etc.

3. **Components CSV** (`ablation_A4_full_components.csv`): Component-level details
   - Shows each claim component's evaluation, gap-filling status, and verdict changes

### Example Metrics Output

```json
{
  "metrics": {
    "accuracy": 0.85,
    "macro_f1": 0.83,
    "per_class": {
      "consistent": {"precision": 0.87, "recall": 0.82, "f1": 0.84},
      "inconsistent": {"precision": 0.81, "recall": 0.86, "f1": 0.83}
    }
  }
}
```

---

## Acknowledgments

This system builds upon the CLAIRE architecture introduced in:

Sina J. Semnani, Jirayu Burapacheep, Arpandeep Khatua, Thanawan Atchariyachanvanit, Zheng Wang, Monica S. Lam. "Detecting Corpus-Level Knowledge Inconsistencies in Wikipedia with Large Language Models." EMNLP 2025.

## License
This code is released under the [Apache 2.0](LICENSE) license.
