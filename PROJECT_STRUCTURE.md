# MLB Live Agent — Project Structure

A Streamlit + Claude-powered scouting tool for MLB pitching analysis. The agent pulls live Statcast data, tracks injuries, runs regressions, and renders interactive Plotly charts — all driven by natural-language queries from the coaching staff.

---

## Root Files

### `MLB_Live_Agent.py`
The Streamlit entry point. Owns the page layout, sidebar controls (team selectors, season, date range), the chat UI, and the agent loop. On each user question it builds an Anthropic API call with prompt caching, dispatches tool calls returned by Claude, converts chart payloads into Plotly figures, and renders them in a 2-column grid above the chat history. This is the only file you run (`streamlit run MLB_Live_Agent.py`).

### `agent_tools.py`
The brain of the agent. Contains every tool function Claude can call, all Plotly chart builders, the ChromaDB helper, the tool registry (`TOOLS` + `TOOL_SCHEMAS`), and the system prompt. Structured so that each tool returns a JSON string with a `text` field (what Claude reads) and optional `chart_type` / `chart_data` fields (what the Streamlit layer renders). Errors stay inside the JSON so Claude can self-correct rather than crashing.

The seven tools defined here are:
- `get_pitcher_statcast` — pitch mix, velocity, xwOBA, barrel rate, whiff rate from Baseball Savant
- `get_il_stints` — IL placements for any team/season via MLB Stats API
- `get_pitch_velocity_trend` — game-by-game fastball velocity trend for fatigue/injury detection
- `get_schedule` — upcoming games and opponents for any team
- `search_pitcher_profile` — semantic search over ChromaDB pitcher summaries
- `build_regression_model` — Ridge regression predicting ERA or xwOBA from Statcast features
- `get_pitch_heatmap_data` — pitch location + movement data for heatmaps and 3D plots

### `migrate_to_duckdb.py`
One-time (and CI-triggered) seed script. Reads `data/Pitchers_Stats.csv` and `data/Batters_Stats.csv`, applies SQL transformations (FIP proxy, whiff tier, within-season normalized contact quality score), and writes `raw_pitchers` and `raw_batters` tables into `mlb_analytics.duckdb`. Run before `dbt build` in the GitHub Actions workflow.

### `mlb_analytics.duckdb`
The local analytics database. Holds the raw and dbt-transformed tables (`raw_pitchers`, `raw_batters`, `stg_pitchers`, `stg_batters`, `mart_pitcher_profiles`). Used by dbt models and queryable directly with the DuckDB CLI. Not used by the live agent loop (which pulls from pybaseball/mlbstatsapi at query time) — it serves the historical analytics layer.

### `requirements.txt`
Single source of truth for all Python dependencies. Covers the data layer (pybaseball, pandas, numpy, mlbstatsapi), the agent layer (anthropic, streamlit, plotly, scikit-learn, chromadb, sentence-transformers), the analytics persistence layer (duckdb, dbt-duckdb), and the live pipeline (beautifulsoup4, lxml, rapidfuzz). Requires Python 3.13.x — chromadb and sentence-transformers don't ship wheels for 3.14+.

---

## `pipeline/`

The live data pipeline that keeps `data/live/` fresh. Run via `python pipeline/run_pipeline.py`. Supports three modes: `full`, `injuries_only`, and `statcast_only`. Has offseason guards (skips Statcast Nov–Feb) and staleness TTL checks (Statcast: 6h, injuries: 1h) so it never re-fetches unnecessarily.

### `pipeline/run_pipeline.py`
The pipeline entrypoint. Parses CLI args, applies TTL and offseason guards, calls the three module functions in order, and appends a structured JSON log entry to `data/live/pipeline_log.jsonl` after every run. A single source failure degrades the run to `"partial"` rather than aborting — the agent always has some data to work with.

### `pipeline/fetch_statcast.py`
Pulls pitcher and batter Statcast data from Baseball Savant via pybaseball for the current and prior season, falling back to Baseball Reference when FanGraphs' legacy leaderboard 403s (which it does consistently from this environment/IP). Writes to `data/live/pitchers_statcast.csv` and `data/live/batters_statcast.csv`.

**Fixed 2026-09-16 (two bugs):**
1. `statsapi_pitcher_exitvelo_barrels()`/`_batter_exitvelo_barrels()` return the player identity as a `'last_name, first_name'` column, not `player_name` — the merge step assumed `player_name` existed and crashed with `KeyError: 'player_name'` on every run that reached it (reproduced during testing; this means the full/statcast_only pipeline modes could never have completed successfully before this fix). Fixed by normalizing the column right after fetch.
2. Even after the crash fix, `xwoba_allowed`/`xwoba` and `hard_hit_pct` were unconditionally null: the code looked for `est_woba` and `hard_hit_percent`/`hard_hit_pct` columns that don't exist on the exitvelo/barrels endpoint at all. Fixed by merging in the separate `statcast_*_expected_stats()` endpoint (which does have `est_woba`) on `player_id`, and by reading the hard-hit rate from its real column name, `ev95percent`. Verified 3x — a ~60-65% null rate on these fields is now expected (Statcast's exitvelo leaderboard only covers *qualified* pitchers/batters, a subset of the full FanGraphs/Baseball-Reference player pool), not a merge failure; the false-alarm "merge likely failed" warning that used to fire at >40% null was retuned to only fire above 85%, since 100% would indicate an actually-broken merge key.

### `pipeline/fetch_injuries.py`
Pulls pitcher IL data from four sources in priority order: the MLB Stats API (structured, most reliable — tried first), then Rotowire, Yahoo Sports, and ESPN as HTML-scraping fallbacks. Caches raw HTML responses to `data/cache/` keyed by source and date so repeat runs within the same day don't re-hit the web. Writes deduped results to `data/live/pitcher_injuries.csv`.

**Fixed 2026-09-16:** the MLB Stats API source pulls every IL transaction league-wide (its transaction payload has no position field), so without filtering it returned batters (Aaron Judge, Mike Trout, Juan Soto, etc.) alongside pitchers — 311 of 647 rows in one test run were non-pitcher noise. Fixed by building a one-call pitcher-ID lookup (`sports_players` endpoint, season-scoped) and filtering MLB API transactions against it before they're written. Verified 3x: unmatched-to-any-pitcher warnings dropped from 311 (all non-pitchers) to 19 (legitimate injured pitchers, mostly with accented names, who simply weren't in the qualified-pitcher Statcast pool that season — expected, not a bug).

### `pipeline/combine.py`
Merges the Statcast and injury CSVs into a unified dataset. Uses rapidfuzz for fuzzy name matching between the two sources (player names differ slightly across data providers). Reports unmatched injury names back to the pipeline log for debugging.

---

## `data/`

Runtime data directory. Not committed to source control (or committed selectively) — regenerated by the pipeline.

### `data/live/pitchers_statcast.csv`
Current-season Statcast aggregates for pitchers. Refreshed by `fetch_statcast.py`.

### `data/live/batters_statcast.csv`
Current-season Statcast aggregates for batters. Refreshed by `fetch_statcast.py`.

### `data/live/pitcher_injuries.csv`
Deduped pitcher IL rows, sourced from the MLB Stats API first and ESPN/Rotowire/Yahoo scraping as fallbacks. Refreshed by `fetch_injuries.py`.

### `data/live/pipeline_log.jsonl`
Append-only JSONL log. Every pipeline run writes one line recording the run timestamp, mode, row counts, duration, status (`success` / `partial` / `error`), and any error messages. Used by `run_pipeline.py` to determine staleness before deciding whether to re-fetch.

---

## `dbt/`

The historical analytics layer. Transforms `raw_pitchers` and `raw_batters` (loaded by `migrate_to_duckdb.py`) into clean staging models and a mart, all inside `mlb_analytics.duckdb`. Run with `dbt build --project-dir dbt --profiles-dir dbt`.

### `dbt/dbt_project.yml`
Project config — name (`mlb_analytics`), model paths, and materialization defaults.

### `dbt/profiles.yml`
DuckDB connection profile. Points dbt at `mlb_analytics.duckdb` in the repo root.

### `dbt/.user.yml`
Auto-generated by dbt. Stores a local anonymous user ID for dbt Cloud telemetry. Safe to ignore.

### `dbt/models/schema.yml`
Column-level documentation and data quality tests for all models — not-null checks, accepted-value checks on `era_regression_flag` and `whiff_tier`. Run automatically by `dbt test`.

### `dbt/models/staging/stg_pitchers.sql`
Selects and lightly cleans `raw_pitchers`. Exposes the columns dbt marts and downstream queries expect: `player_id`, `season`, `p_era`, `xwoba`, `whiff_percent`, `whiff_tier`, FIP proxy, etc.

### `dbt/models/staging/stg_batters.sql`
Same pattern for batters. Exposes `player_id`, `season`, `iso`, `contact_quality_score`, and the normalized barrel/sweet-spot/exit-velo metrics computed in `migrate_to_duckdb.py`.

### `dbt/models/marts/mart_pitcher_profiles.sql`
The primary analytical mart. Joins `stg_pitchers` with aggregate batter contact quality to produce a per-pitcher profile row. Adds `era_minus_fip` and an `era_regression_flag` (`1` when ERA is at least 0.50 above the FIP proxy with 20+ IP — a sign of likely ERA regression; until 2026-10-21 the SQL compared ERA with xwOBA by mistake) and surfaces the metrics most useful for scouting reports.

---

## `.github/workflows/`

### `.github/workflows/refresh_data.yml`
GitHub Actions workflow that runs every Sunday at 6 AM UTC (plus manual trigger). Sets up Python 3.13, installs dependencies, seeds DuckDB from CSVs (`migrate_to_duckdb.py`), runs `dbt build`, prints a row-count summary, and runs `dbt test`. Keeps the historical analytics layer current without any manual intervention.


---

## Testing status (2026-09-16)

Verified by actually running the code, 3x each for reliability, not just read through:

- `pipeline/run_pipeline.py --mode injuries_only` — 3/3 clean runs, consistent row counts (775 pitcher IL rows → 330 deduped, 19 legitimate unmatched).
- `pipeline/run_pipeline.py --mode statcast_only` — 3/3 clean runs after fixing the two bugs in `fetch_statcast.py` above (previously crashed every time it reached the merge step).
- `pipeline/run_pipeline.py --mode full` — 3/3 clean runs, `status=success`, 1724 pitcher rows / 1334 batter rows / 330 injury rows / 1724 combined.
- `dbt run` and `dbt test` against the existing `mlb_analytics.duckdb` — 3/3 clean runs, 3/3 models build, 13/13 data tests pass.

Not testable in this pass, and not faked:

- `migrate_to_duckdb.py` needs `data/Pitchers_Stats.csv` and `data/Batters_Stats.csv`, which aren't present in this copy of the repo (likely excluded from the working folder or never committed). The existing `mlb_analytics.duckdb` already has `raw_pitchers`/`raw_batters` built from a prior run, which is what let the dbt layer above get tested — but the migration script's CSV-loading SQL itself was reviewed, not executed. Add the two CSVs back to `data/` and re-run `migrate_to_duckdb.py` to confirm it end-to-end.
- The live Streamlit *chat loop* (`MLB_Live_Agent.py`'s Claude conversation handling) requires `ANTHROPIC_API_KEY` and a running Streamlit session — neither is available in this environment, so that specific wiring (tool-call dispatch, chart rendering into the Streamlit UI) is read-reviewed but not exercised end-to-end.

However, all seven tool functions in `agent_tools.py` themselves — the actual logic Claude would call — were run directly against live data (not mocked), 3x each. This surfaced and fixed five real, reproducible bugs:

1. **`tool_get_pitcher_statcast` — barrel rate always `None`.** The code looked for a `barrel` column that doesn't exist anywhere in `statcast_pitcher()`'s pitch-level output. Fixed to use the real per-pitch signal, `launch_speed_angle == 6` (Statcast's own barrel code), as barrels ÷ batted-ball events.
2. **`tool_get_il_stints` — 100% broken, every call.** Imported `mlbstatsapi`, which is not an installable PyPI package (confirmed 404 on pypi.org). Rewritten to use `statsapi` (`MLB-StatsAPI`), the package already relied on elsewhere in this repo and the one actually in `requirements.txt`'s intent.
3. **`tool_get_schedule` — same fictitious-package bug**, same fix.
4. **`tool_build_regression` — three stacked bugs**: no fallback when FanGraphs' leaderboard 403s (which it does reliably from this network, so the tool always errored); `FEATURE_MAP` pointed `barrel_pct`/`hard_hit_percent`/`exit_velocity_avg`/`sweet_spot_percent` at Statcast column names that don't exist (real names are `brl_percent`/`ev95percent`/`avg_hit_speed`/`anglesweetspotpercent`); and even after those fixes, the Statcast merge key didn't reformat Statcast's `"Last, First"` names to FanGraphs' `"First Last"` order, so it silently matched 0 of 578 rows. All four fixed; verified 3x with a real 532-pitcher regression (ERA ~ K% + BB% + FIP + avg exit velo, R²=0.492).
5. **`requirements.txt`** listed `mlbstatsapi>=12.0.0`, the same nonexistent package from bug #2/#3 — would have failed `pip install -r requirements.txt` on a completely fresh clone. Corrected to `MLB-StatsAPI`.

`tool_get_velocity_trend`, `tool_search_pitcher_profile`, and `tool_get_pitch_heatmap` were also run 3x each and worked correctly with no changes needed.
