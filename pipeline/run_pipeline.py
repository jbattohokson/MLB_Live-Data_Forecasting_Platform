"""
run_pipeline.py — Entrypoint for the MLB Live Agent data pipeline.

Usage:
    python pipeline/run_pipeline.py [--mode full|injuries_only|statcast_only]

Modes:
    full (default)    — fetch_statcast → fetch_injuries → combine
    injuries_only     — fetch_injuries → combine  (Statcast skipped)
    statcast_only     — fetch_statcast → combine  (injury scrape skipped)

After every run, one JSON line is appended to data/live/pipeline_log.jsonl.
A single source failure degrades the run to "partial" — it never aborts entirely.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
import traceback
from pathlib import Path

# Ensure pipeline/ is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))

import combine
import fetch_injuries
import fetch_statcast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("run_pipeline")

ROOT     = Path(__file__).resolve().parent.parent
LIVE_DIR = ROOT / "data" / "live"
LOG_FILE = LIVE_DIR / "pipeline_log.jsonl"

STATCAST_TTL_HOURS = 6
INJURY_TTL_HOURS   = 1


def _last_pull_age_hours(mode_tag: str) -> float | None:
    """Return hours since the last successful pull of mode_tag, or None if no log entry."""
    if not LOG_FILE.exists():
        return None
    try:
        entries = [json.loads(line) for line in LOG_FILE.read_text().splitlines() if line.strip()]
        matching = [
            e for e in entries
            if e.get("status") in ("success", "partial") and mode_tag in e.get("mode", "")
        ]
        if not matching:
            return None
        last_ts = matching[-1].get("run_at", "")
        last_dt = datetime.datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        delta   = datetime.datetime.now(datetime.timezone.utc) - last_dt
        return delta.total_seconds() / 3600
    except Exception as exc:
        log.warning("Could not parse pipeline log: %s", exc)
        return None


def _is_offseason() -> bool:
    """True if today falls in the MLB offseason (November 1 – February 28/29)."""
    today = datetime.date.today()
    return today.month in (11, 12, 1, 2)


def _append_log(entry: dict) -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _run_statcast(years: list[int], stats: dict, errors: list[str]) -> None:
    """Run fetch_statcast.run() and merge its stats into the shared dict."""
    try:
        result = fetch_statcast.run(years)
        stats["statcast_pitcher_rows"] = result.get("statcast_pitcher_rows", 0)
        stats["statcast_batter_rows"]  = result.get("statcast_batter_rows", 0)
        errors.extend(result.get("errors", []))
    except Exception as exc:
        msg = f"fetch_statcast crashed: {traceback.format_exc(limit=3)}"
        log.error(msg)
        errors.append(msg)


def _run_injuries(year: int, stats: dict, errors: list[str]) -> None:
    """Run fetch_injuries.run() and merge its stats into the shared dict."""
    try:
        result = fetch_injuries.run(year)
        stats["injury_rows"] = result.get("injury_rows", 0)
        errors.extend(result.get("errors", []))
    except Exception as exc:
        msg = f"fetch_injuries crashed: {traceback.format_exc(limit=3)}"
        log.error(msg)
        errors.append(msg)


def _run_combine(stats: dict, errors: list[str]) -> None:
    """Run combine.run() and merge its stats into the shared dict."""
    try:
        result = combine.run()
        stats["combined_rows"]          = result.get("combined_rows", 0)
        stats["unmatched_injury_names"] = result.get("unmatched_injury_names", [])
        errors.extend(result.get("errors", []))
    except Exception as exc:
        msg = f"combine crashed: {traceback.format_exc(limit=3)}"
        log.error(msg)
        errors.append(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLB Live Agent data pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "injuries_only", "statcast_only"],
        default="full",
        help="Which data sources to refresh (default: full)",
    )
    args = parser.parse_args()

    today       = datetime.date.today()
    this_year   = today.year
    prior_year  = this_year - 1
    years       = [prior_year, this_year]

    t0     = time.perf_counter()
    stats:  dict       = {"mode": args.mode, "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    errors: list[str]  = []

    run_statcast  = args.mode in ("full", "statcast_only")
    run_injuries  = args.mode in ("full", "injuries_only")

    # Offseason guard — Statcast data doesn't update Nov–Feb
    if run_statcast and _is_offseason():
        log.info("Offseason detected (month=%d) — Statcast pull skipped", today.month)
        stats["statcast_pitcher_rows"] = 0
        stats["statcast_batter_rows"]  = 0
        run_statcast = False

    # Staleness checks — skip re-fetch within TTL windows
    if run_statcast:
        age = _last_pull_age_hours(args.mode)
        if age is not None and age < STATCAST_TTL_HOURS:
            log.info("Statcast data is %.1fh old (TTL=%dh) — skipping fetch", age, STATCAST_TTL_HOURS)
            run_statcast = False

    if run_injuries:
        age = _last_pull_age_hours("injuries")
        if age is not None and age < INJURY_TTL_HOURS:
            log.info("Injury data is %.1fh old (TTL=%dh) — skipping fetch", age, INJURY_TTL_HOURS)
            run_injuries = False

    # Execute pipeline steps
    if run_statcast:
        log.info("Fetching Statcast + FanGraphs for years: %s", years)
        _run_statcast(years, stats, errors)

    if run_injuries:
        log.info("Fetching pitcher injury reports")
        _run_injuries(this_year, stats, errors)

    log.info("Running combine")
    _run_combine(stats, errors)

    # Determine overall status
    has_data   = stats.get("statcast_pitcher_rows", 0) > 0 or stats.get("injury_rows", 0) > 0
    status     = "success" if not errors else ("partial" if has_data else "error")
    duration   = round(time.perf_counter() - t0, 1)

    log_entry = {
        "run_at":                stats["run_at"],
        "mode":                  args.mode,
        "statcast_pitcher_rows": stats.get("statcast_pitcher_rows", 0),
        "statcast_batter_rows":  stats.get("statcast_batter_rows", 0),
        "injury_rows":           stats.get("injury_rows", 0),
        "combined_rows":         stats.get("combined_rows", 0),
        "unmatched_injury_names": stats.get("unmatched_injury_names", []),
        "duration_seconds":      duration,
        "status":                status,
        "errors":                errors,
    }
    _append_log(log_entry)

    log.info(
        "Run complete in %.1fs — status=%s  "
        "statcast_pitcher_rows_this_run=%d  statcast_batter_rows_this_run=%d  "
        "injury_rows_this_run=%d  combined_rows_in_file=%d",
        duration, status,
        log_entry["statcast_pitcher_rows"],
        log_entry["statcast_batter_rows"],
        log_entry["injury_rows"],
        log_entry["combined_rows"],
    )
    if errors:
        log.warning("Errors logged (%d): %s", len(errors), errors)


if __name__ == "__main__":
    main()
