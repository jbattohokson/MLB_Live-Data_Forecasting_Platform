# MLB Live Agent

A Streamlit app where a coach or analyst asks pitching questions in plain English ("how has Player X's fastball velocity trended over his last five starts, and is that a fatigue signal?") and a Claude-powered agent pulls live Statcast and injury data, runs the analysis, and renders the answer as an interactive Plotly chart.

**Business question this answers:** can a natural-language interface over live sports data shorten the time between "I have a question" and "I have a chart" for a non-technical end user (a coach), instead of requiring them to file a request with an analyst?

## Architecture

- **`MLB_Live_Agent.py`** — Streamlit entry point: page layout, sidebar controls, chat UI, and the agent loop (send question, parse Claude's tool calls, dispatch them, render results).
- **`agent_tools.py`** — the seven tools Claude can call: live Statcast pitch metrics, IL stints, velocity trends, schedules, semantic search over pitcher profiles (ChromaDB), a Ridge regression model predicting ERA/xwOBA from Statcast features, and pitch-location heatmap data.
- **`pipeline/`** — the live data refresh pipeline (`run_pipeline.py` orchestrates `fetch_statcast.py`, `fetch_injuries.py`, `combine.py`), with offseason guards and staleness TTLs so it doesn't re-fetch unnecessarily. Runs on a schedule via `.github/workflows/refresh_data.yml`.
- **`migrate_to_duckdb.py`** + **`dbt/`** — the historical analytics layer: seeds DuckDB from season-level stats CSVs, then dbt staging and mart models compute FIP proxies, whiff tiers, and normalized contact-quality scores. This layer is separate from the live agent loop, which queries pybaseball/MLB Stats API directly at question time.

See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for a file-by-file breakdown, plus a full record of what was tested and fixed on 2026-09-16.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run MLB_Live_Agent.py
```

## Tools

Python, Streamlit, DuckDB, dbt, ChromaDB, Anthropic Claude API (agentic tool use), Plotly, scikit-learn, pybaseball, MLB Stats API, GitHub Actions.

## Testing status (2026-09-16)

Every tool function and pipeline stage was actually run against live data — 3x each for reliability, not just read through. This surfaced and fixed real bugs, not cosmetic ones: a crash in the Statcast merge that meant the full pipeline could never complete successfully, an always-null xwOBA/hard-hit-rate field caused by reading the wrong column names, non-pitchers polluting the injury feed, and two of the seven agent tools (`get_il_stints`, `get_schedule`) that were 100% broken because they imported a Python package that doesn't exist on PyPI. Full details, including exact before/after numbers, are in [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md#testing-status-2026-09-16).

Two things couldn't be tested end-to-end in this environment and are documented rather than assumed: `migrate_to_duckdb.py` (the source CSVs aren't present in this copy of the repo — the dbt layer was still verified against the already-built database) and the live Claude chat loop itself (no `ANTHROPIC_API_KEY` available here — the seven underlying tool functions it calls were fully tested instead).

## Live demo

*(This is the single highest-leverage thing to finish before applications: deploy to Streamlit Community Cloud — free — and put the live URL here and at the top of the resume bullet. A working link beats a GitHub README every time a recruiter has 30 seconds.)*

## Repo hygiene note

`mlb_analytics.duckdb`, `data/live/`, `data/cache/`, and `__pycache__/` are excluded from this repo via `.gitignore` — they're build/runtime artifacts, not source. `migrate_to_duckdb.py` regenerates the database from the raw CSVs, and `pipeline/run_pipeline.py` regenerates the live data, so nothing is lost by not committing them.
