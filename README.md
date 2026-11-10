# LLM Digital Twin

A quick-start toy project for simulating human survey responses with an LLM. Each text file in `prompt/` describes a person's earlier survey answers and asks the model to answer a new 60-item questionnaire in that person's voice. The project calls a DashScope-compatible chat API, collects the model's answers, and compares them with the matching human responses.

This is a small demonstration, not a validated digital twin or a research-grade evaluation. The included sample has eight prompt IDs.

## What is included

| Path | Purpose |
| --- | --- |
| `prompt/pid_<id>_prompt.txt` | One persona and new questionnaire per ID |
| `1_run_llm.py` | Submit prompts and save one response per ID in `json/` |
| `2_processjson.py` | Convert JSON responses into `llm_answers.csv` |
| `3_metrics.py` | Match model and human answers by ID and print summary metrics |
| `humanresponse.csv` | Human answers for the eight included prompt IDs |
| `json/`, `llm_answers.csv` | Example outputs from a previous run |

## Quick start

Requires Python 3.10 or newer and a DashScope-compatible API key. Commands below run from the project directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your key in the local `.env` file:

```dotenv
DASHSCOPE_API_KEY=your_key_here
```

Then run:

```bash
python 1_run_llm.py
python 2_processjson.py
python 3_metrics.py
```

`1_run_llm.py` uses the model name and request settings defined near the top of the script. Check that `MODEL_NAME` is available to your account and change it if needed. Existing `json/<id>.json` files are skipped by default (`SKIP_EXISTING = True`), so a fresh API run requires removing the corresponding example output files or setting `SKIP_EXISTING = False`. API calls may incur charges.

To inspect the included example without an API key, install the dependencies and run only `python 3_metrics.py`. You can also regenerate `llm_answers.csv` from the included JSON files with `python 2_processjson.py`.

## Data and evaluation

The prompt filename supplies the ID. Model answers are expected to be a JSON object with string keys `"1"` through `"60"` and integer values from 1 to 5. `2_processjson.py` extracts those answers into a CSV row. `3_metrics.py` joins that CSV to `humanresponse.csv` using `id` and `CVDID`, respectively, and reports mean response accuracy, item-level and profile Spearman correlations, and error slopes with 95% confidence intervals.

The sample contains only eight people. Treat its metrics as a pipeline demonstration; they do not establish predictive validity or generalize to a population. The prompts contain survey profiles, so review any replacement data for consent and privacy before publishing it.

## Repository hygiene

The local `.env` file is ignored by Git. Keep API keys there or in your shell environment, and never commit credentials. The provided `.env.example` contains only an empty setting.
