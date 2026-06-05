# MLB Live Agent

A Streamlit web application powered by a Claude agentic loop for MLB pitching analysis. Ask natural-language questions about any pitcher or team — the agent pulls live Statcast data, tracks injury history, runs Ridge regression, and renders interactive Plotly charts directly in the chat interface.

> **Status:** Active development. Core agent loop, all 7 tools, dbt analytics layer, and live pipeline are functional. Additional chart types and batter-side tooling in progress.

---

## What It Does

Traditional scouting workflows rely on static reports that can't respond to ad-hoc questions or surface regression signals in real time. This agent replaces that workflow with a conversational interface backed by live data.

Example queries:
- "What's Zack Wheeler's pitch mix and xwOBA allowed this season?"
- "Show me the Dodgers' IL history for 2024."
- "Build a regression model predicting ERA from whiff rate and barrel rate."
- "Plot Corbin Burnes' fastball velocity trend over the last 30 days."
- "Which Phillies pitchers are showing velocity decline this month?"

---

## Architecture

```
MLB_Live_Agent.py          Streamlit entry point. Owns page layout, sidebar controls,
                           chat UI, and the Claude agentic loop. Dispatches tool calls,
                           converts chart payloads into Plotly figures, renders them
                           in a 2-column grid above chat history.

agent_tools.py             All 7 tool functions, Plotly chart builders, ChromaDB helper,
                           tool registry (TOOLS + TOOL_SCHEMAS), and system prompt.
                           Each tool returns a JSON string with a "text" field Claude
                           reads and optional chart_type / chart_data fields the
                           Streamlit layer renders. Errors stay inside JSON so Claude
                           can self-correct and retry without crashing.

pipeline/                  Live data pipeline keeping data/live/ fresh.
  run_pipeline.py          Entrypoint. Supports full, injuries_only, statcast_only modes.
                           Applies TTL staleness checks (Statcast: 6h, injuries: 1h)
                           and offseason guards (skips Statcast Nov-Feb).
  fetch_statcast.py        Pulls pitcher and batter Statcast data from Baseball Savant
                           via pybaseball. Writes to data/live/pitchers_statcast.csv.
  fetch_injuries.py        Scrapes pitcher IL reports from ESPN, Rotowire, and Yahoo.
                           Caches raw HTML by source and date. Writes deduped results
                           to data/live/pitcher_injuries.csv.
  combine.py               Merges Statcast and injury CSVs using rapidfuzz for fuzzy
                           name matching across data providers.

migrate_to_duckdb.py       Seeds mlb_analytics.duckdb from historical CSVs. Applies
                           SQL transforms: FIP proxy, whiff tier, normalized contact
                           quality score. Run before dbt build.

dbt/                       Historical analytics layer inside mlb_analytics.duckdb.
  staging/stg_pitchers     Cleans raw_pitchers, exposes ERA, xwOBA, whiff_tier, FIP.
  staging/stg_batters      Cleans raw_batters, exposes ISO, contact_quality_score.
  marts/mart_pitcher_profiles  Per-pitcher profile rows with era_regression_flag
                           (1 when ERA materially exceeds FIP — likely regression).

.github/workflows/
  refresh_data.yml         GitHub Actions workflow. Runs every Sunday at 6 AM UTC.
                           Seeds DuckDB, runs dbt build, prints row-count summary,
                           runs dbt test. Keeps the analytics layer current without
                           manual intervention.
```

---

## Agent Tools

| Tool | Data Source | What It Does |
|---|---|---|
| `get_pitcher_statcast` | Baseball Savant / pybaseball | Pitch mix, velocity, xwOBA, barrel rate, whiff rate |
| `get_il_stints` | MLB Stats API | IL placements for any team and season |
| `get_pitch_velocity_trend` | Baseball Savant | Game-by-game fastball velocity for fatigue/injury detection |
| `get_schedule` | MLB Stats API | Upcoming games and opponents for any team |
| `search_pitcher_profile` | ChromaDB | Semantic search over pitcher summaries |
| `build_regression_model` | pybaseball + scikit-learn | Ridge regression predicting ERA or xwOBA from Statcast features |
| `get_pitch_heatmap_data` | Baseball Savant | Pitch location and movement for heatmaps and 3D plots |

---

## Stack

- **Languages:** Python 3.13
- **UI:** Streamlit
- **Agent:** Anthropic Claude API (tool use + prompt caching)
- **Live data:** pybaseball, mlbstatsapi
- **Analytics DB:** DuckDB + dbt
- **Vector store:** ChromaDB + sentence-transformers
- **Modeling:** scikit-learn (Ridge regression)
- **Visualization:** Plotly
- **Pipeline:** GitHub Actions (weekly refresh)
- **Other:** rapidfuzz, beautifulsoup4, pandas, NumPy

---

## Setup & Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Seed the analytics database (run once, then automated via GitHub Actions)
python migrate_to_duckdb.py
dbt build --project-dir dbt --profiles-dir dbt

# Run the live data pipeline
python pipeline/run_pipeline.py --mode full

# Launch the app
streamlit run MLB_Live_Agent.py
```

> **Python version:** Requires Python 3.13.x. chromadb and sentence-transformers do not ship binary wheels for Python 3.14+.

---

## Data Layers

The agent separates live and historical data intentionally:

**Live layer** (`data/live/`) — refreshed by the pipeline on demand. Used by the agent at query time for current-season Statcast stats and injury reports.

**Historical layer** (`mlb_analytics.duckdb`) — seeded from `Datasets/` CSVs and transformed by dbt. Used for multi-season trend analysis and regression modeling. Not used by the live agent loop directly.

**Vector store** (ChromaDB) — pitcher summaries embedded at ingest time. Enables semantic search for profile-style questions that don't map cleanly to structured queries.
