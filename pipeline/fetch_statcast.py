"""
fetch_statcast.py — Pull FanGraphs pitching/batting stats and Statcast exit-velo
data for the current and prior season. Merge on normalized player name and write
atomic CSVs to data/live/. Every external call has a failure path.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent.parent
LIVE_DIR  = ROOT / "data" / "live"
CACHE_DIR = ROOT / "data" / "cache"

# Canonical output columns — order matters for the CSV header.
PITCHER_COLS = [
    "player_name", "season", "team", "ip", "era", "fip", "xfip",
    "k_pct", "bb_pct", "babip", "lob_pct", "hr9",
    "xwoba_allowed", "barrel_rate_allowed", "exit_velocity_avg",
    "hard_hit_pct", "whiff_pct", "data_source", "pulled_at",
]
BATTER_COLS = [
    "player_name", "season", "team", "pa", "avg", "obp", "slg", "ops",
    "woba", "xwoba", "barrel_pct", "exit_velocity_avg", "hard_hit_pct",
    "k_pct", "bb_pct", "data_source", "pulled_at",
]


def _normalize_name(name: str) -> str:
    """Normalize 'Last, First' or 'First Last' to 'first last' for merge key.

    Statcast returns 'Last, First'; FanGraphs returns 'First Last'. Without this,
    the merge key mismatch silently produces a full outer join of nulls.
    """
    name = name.strip()
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        return f"{parts[1]} {parts[0]}".lower()
    return name.lower()


def _write_atomic(df: pd.DataFrame, target: Path) -> None:
    """Write to .tmp then rename — guards against partial writes corrupting live CSVs."""
    df = df.sort_values(
        ["season", "player_name"], ascending=[False, True]
    ).reset_index(drop=True)
    tmp = target.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.rename(target)


def _sc_get(row: pd.Series, *keys: str) -> Any:
    """Try multiple Statcast column name variants — pybaseball column names shift between versions."""
    for k in keys:
        val = row.get(k)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            return val
    return None


def _normalize_bref_pitching(df: pd.DataFrame) -> pd.DataFrame:
    """Remap Baseball Reference column names to match the FanGraphs schema."""
    rename = {"Tm": "Team", "SO9": "_so9", "BB9": "_bb9", "HR9": "HR/9"}
    df = df.rename(columns=rename).copy()
    # Approximate K% and BB% from rate stats when BF is available
    if "BF" in df.columns and "SO" in df.columns:
        df["K%"] = df["SO"] / df["BF"].replace(0, float("nan"))
    if "BF" in df.columns and "BB" in df.columns:
        df["BB%"] = df["BB"] / df["BF"].replace(0, float("nan"))
    for col in ("xFIP", "LOB%", "BABIP"):
        if col not in df.columns:
            df[col] = float("nan")
    return df


def _add_player_name_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a `player_name` column exists on a Statcast leaderboard frame.

    pybaseball's statcast_*_exitvelo_barrels() returns the player identity as
    `'last_name, first_name'` (e.g. "Irvin, Jake"), not `player_name`. Every
    downstream merge keys off `player_name`, so without this the merge crashes
    with KeyError: 'player_name' — which means historically this step has
    never actually merged Statcast data into the pitcher/batter rows; FanGraphs
    data was silently used alone whenever this crash was swallowed upstream,
    or the whole run errored out (as reproduced during testing on 2026-09-16).
    """
    if "player_name" in df.columns:
        return df
    if "last_name, first_name" in df.columns:
        df = df.copy()
        df["player_name"] = df["last_name, first_name"].astype(str)
        return df
    log.warning(
        "Statcast frame has neither 'player_name' nor 'last_name, first_name' "
        "column (columns=%s) — merge will find no matches for these rows",
        list(df.columns),
    )
    return df


def _fetch_fg_pitching(year: int) -> pd.DataFrame | None:
    """Pull FanGraphs pitching stats; fall back to Baseball Reference on 403."""
    import pybaseball  # type: ignore[import-untyped]
    pybaseball.cache.enable()

    # Primary: FanGraphs (may return 403 — pybaseball scrapes a legacy endpoint)
    try:
        df = pybaseball.pitching_stats(year, year, qual=20)
        if df is not None and not df.empty:
            return df
        log.warning("FanGraphs pitching returned empty for %d", year)
    except Exception as exc:
        log.warning("FanGraphs pitching %d unavailable (%s) — trying Baseball Reference", year, exc)

    # Fallback: Baseball Reference (open, no auth required)
    try:
        df = pybaseball.pitching_stats_bref(year)
        if df is not None and not df.empty:
            log.info("Using Baseball Reference pitching data for %d", year)
            return _normalize_bref_pitching(df)
        log.error("Baseball Reference pitching also empty for %d", year)
    except Exception as exc:
        log.error("Baseball Reference pitching %d failed: %s", year, exc)
    return None


def _fetch_sc_pitching(year: int) -> pd.DataFrame | None:
    """Pull Statcast pitcher exit-velo/barrel data, merged with expected stats.

    statcast_pitcher_exitvelo_barrels() alone has no wOBA-against field at all
    (confirmed by inspecting its real columns during testing on 2026-09-16) —
    the `est_woba` field this pipeline needs lives on the separate
    statcast_pitcher_expected_stats() endpoint. Without this merge,
    xwoba_allowed was unconditionally null for every row, every run.
    """
    try:
        import pybaseball  # type: ignore[import-untyped]
        # minBBE='q' (qualified) is the correct parameter name and default;
        # older calls used minPA which does not exist in pybaseball's signature.
        df = pybaseball.statcast_pitcher_exitvelo_barrels(year)
        if df is None or df.empty:
            log.warning("Statcast pitcher exitvelo empty for %d", year)
            return None
        df = _add_player_name_col(df)

        try:
            exp = pybaseball.statcast_pitcher_expected_stats(year)
            if exp is not None and not exp.empty and "player_id" in exp.columns:
                exp = exp[["player_id", "est_woba"]].drop_duplicates("player_id")
                df = df.merge(exp, on="player_id", how="left")
            else:
                log.warning("Statcast pitcher expected_stats empty/malformed for %d", year)
        except Exception as exc:
            log.warning("Statcast pitcher expected_stats %d failed: %s — xwoba_allowed will be null", year, exc)

        return df
    except Exception as exc:
        log.error("Statcast pitcher exitvelo %d failed: %s", year, exc)
        return None


def _build_pitcher_row(fg: pd.Series, sc: pd.Series | None, year: int) -> dict:
    """Map FanGraphs + Statcast columns to the canonical pitcher output schema."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    s   = sc if sc is not None else pd.Series(dtype=object)
    return {
        "player_name":         fg.get("Name", ""),
        "season":              year,
        "team":                fg.get("Team", ""),
        "ip":                  fg.get("IP"),
        "era":                 fg.get("ERA"),
        "fip":                 fg.get("FIP"),
        "xfip":                fg.get("xFIP"),
        "k_pct":               fg.get("K%"),
        "bb_pct":              fg.get("BB%"),
        "babip":               fg.get("BABIP"),
        "lob_pct":             fg.get("LOB%"),
        "hr9":                 fg.get("HR/9"),
        "xwoba_allowed":       _sc_get(s, "est_woba", "xwoba"),
        "barrel_rate_allowed": _sc_get(s, "brl_percent", "barrel_batted_rate"),
        "exit_velocity_avg":   _sc_get(s, "avg_hit_spd", "avg_hit_speed", "avg_exit_velocity"),
        "hard_hit_pct":        _sc_get(s, "hard_hit_percent", "hard_hit_pct", "ev95percent"),
        "whiff_pct":           _sc_get(s, "whiff_percent") or fg.get("SwStr%") or fg.get("Swing-Strike%"),
        "data_source":         "pybaseball",
        "pulled_at":           now,
    }


def _merge_pitching(fg: pd.DataFrame, sc: pd.DataFrame | None, year: int) -> pd.DataFrame:
    """Join FanGraphs and Statcast pitcher data on normalized name key."""
    fg = fg.copy()
    fg["_key"] = fg["Name"].astype(str).map(_normalize_name)

    sc_index: dict[str, pd.Series] = {}
    if sc is not None and not sc.empty:
        sc = sc.copy()
        sc["_key"] = sc["player_name"].astype(str).map(_normalize_name)
        sc_index   = {row["_key"]: row for _, row in sc.iterrows()}

    rows = [_build_pitcher_row(r, sc_index.get(r["_key"]), year) for _, r in fg.iterrows()]
    df   = pd.DataFrame(rows)

    # Melchior data-integrity checks
    if len(df) < 400:
        log.warning("Pitcher rows %d below threshold 400 for %d — qual filter may be too strict", len(df), year)
    # Statcast's exitvelo/expected-stats leaderboards only cover *qualified*
    # pitchers (min batted-ball-event threshold), so a 60-65% null rate here is
    # normal coverage, not a failed merge — confirmed against a known-good run
    # during testing on 2026-09-16. Only flag when it's high enough to indicate
    # the merge key itself broke (e.g. every row null).
    xwoba_null = df["xwoba_allowed"].isna().mean()
    if xwoba_null > 0.85:
        log.warning(
            "xwoba_allowed null rate %.0f%% (expected ~60-65%% from Statcast's "
            "qualified-pitcher coverage) — this high suggests the merge key "
            "itself is broken for %d, not just normal coverage limits",
            xwoba_null * 100, year,
        )
    era_null = df["era"].isna().mean()
    if era_null > 0.05:
        log.error("ERA null rate %.0f%% — FanGraphs pull may be broken for %d", era_null * 100, year)

    dupes = df.duplicated(["player_name", "season"]).sum()
    if dupes:
        log.warning("Dropping %d duplicate player-season rows for %d (keeping highest IP)", dupes, year)
        df = df.sort_values("ip", ascending=False).drop_duplicates(["player_name", "season"])

    return df.reindex(columns=PITCHER_COLS)


def _fetch_fg_batting(year: int) -> pd.DataFrame | None:
    """Pull FanGraphs batting stats; fall back to Baseball Reference on 403."""
    import pybaseball  # type: ignore[import-untyped]
    pybaseball.cache.enable()

    try:
        df = pybaseball.batting_stats(year, year, qual=50)
        if df is not None and not df.empty:
            return df
        log.warning("FanGraphs batting returned empty for %d", year)
    except Exception as exc:
        log.warning("FanGraphs batting %d unavailable (%s) — trying Baseball Reference", year, exc)

    try:
        df = pybaseball.batting_stats_bref(year)
        if df is not None and not df.empty:
            log.info("Using Baseball Reference batting data for %d", year)
            # BRef uses Tm for team; wOBA and xwOBA not available — leave as NaN
            return df.rename(columns={"Tm": "Team"}).copy()
        log.error("Baseball Reference batting also empty for %d", year)
    except Exception as exc:
        log.error("Baseball Reference batting %d failed: %s", year, exc)
    return None


def _fetch_sc_batting(year: int) -> pd.DataFrame | None:
    """Pull Statcast batter exit-velo/barrel data, merged with expected stats.

    Same fix as _fetch_sc_pitching: exitvelo_barrels has no `est_woba` column,
    so xwoba was unconditionally null without this merge.
    """
    try:
        import pybaseball  # type: ignore[import-untyped]
        # minBBE='q' (qualified) is the correct parameter name and default;
        # older calls used minPA which does not exist in pybaseball's signature.
        df = pybaseball.statcast_batter_exitvelo_barrels(year)
        if df is None or df.empty:
            log.warning("Statcast batter exitvelo empty for %d", year)
            return None
        df = _add_player_name_col(df)

        try:
            exp = pybaseball.statcast_batter_expected_stats(year)
            if exp is not None and not exp.empty and "player_id" in exp.columns:
                exp = exp[["player_id", "est_woba"]].drop_duplicates("player_id")
                df = df.merge(exp, on="player_id", how="left")
            else:
                log.warning("Statcast batter expected_stats empty/malformed for %d", year)
        except Exception as exc:
            log.warning("Statcast batter expected_stats %d failed: %s — xwoba will be null", year, exc)

        return df
    except Exception as exc:
        log.error("Statcast batter exitvelo %d failed: %s", year, exc)
        return None


def _build_batter_row(fg: pd.Series, sc: pd.Series | None, year: int) -> dict:
    """Map FanGraphs + Statcast columns to the canonical batter output schema."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    s   = sc if sc is not None else pd.Series(dtype=object)
    return {
        "player_name":       fg.get("Name", ""),
        "season":            year,
        "team":              fg.get("Team", ""),
        "pa":                fg.get("PA"),
        "avg":               fg.get("AVG"),
        "obp":               fg.get("OBP"),
        "slg":               fg.get("SLG"),
        "ops":               fg.get("OPS"),
        "woba":              fg.get("wOBA"),
        "xwoba":             _sc_get(s, "est_woba", "xwoba"),
        "barrel_pct":        _sc_get(s, "brl_percent", "barrel_batted_rate"),
        "exit_velocity_avg": _sc_get(s, "avg_hit_spd", "avg_hit_speed", "avg_exit_velocity"),
        "hard_hit_pct":      _sc_get(s, "hard_hit_percent", "hard_hit_pct", "ev95percent"),
        "k_pct":             fg.get("K%"),
        "bb_pct":            fg.get("BB%"),
        "data_source":       "pybaseball",
        "pulled_at":         now,
    }


def _merge_batting(fg: pd.DataFrame, sc: pd.DataFrame | None, year: int) -> pd.DataFrame:
    """Join FanGraphs and Statcast batter data on normalized name key."""
    fg = fg.copy()
    fg["_key"] = fg["Name"].astype(str).map(_normalize_name)

    sc_index: dict[str, pd.Series] = {}
    if sc is not None and not sc.empty:
        sc = sc.copy()
        sc["_key"] = sc["player_name"].astype(str).map(_normalize_name)
        sc_index   = {row["_key"]: row for _, row in sc.iterrows()}

    rows = [_build_batter_row(r, sc_index.get(r["_key"]), year) for _, r in fg.iterrows()]
    df   = pd.DataFrame(rows)

    if len(df) < 900:
        log.warning("Batter rows %d below threshold 900 for %d", len(df), year)

    dupes = df.duplicated(["player_name", "season"]).sum()
    if dupes:
        log.warning("Dropping %d duplicate batter rows for %d", dupes, year)
        df = df.sort_values("pa", ascending=False).drop_duplicates(["player_name", "season"])

    return df.reindex(columns=BATTER_COLS)


def run(years: list[int]) -> dict[str, Any]:
    """Pull Statcast + FanGraphs for all requested years. Returns log stats dict."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_pitchers: list[pd.DataFrame] = []
    all_batters:  list[pd.DataFrame] = []
    errors:       list[str]          = []

    for year in years:
        fg_p = _fetch_fg_pitching(year)
        if fg_p is not None:
            sc_p = _fetch_sc_pitching(year)
            all_pitchers.append(_merge_pitching(fg_p, sc_p, year))
        else:
            errors.append(f"FanGraphs pitching unavailable for {year}")

        fg_b = _fetch_fg_batting(year)
        if fg_b is not None:
            sc_b = _fetch_sc_batting(year)
            all_batters.append(_merge_batting(fg_b, sc_b, year))
        else:
            errors.append(f"FanGraphs batting unavailable for {year}")

    pitcher_rows = batter_rows = 0

    if all_pitchers:
        combined = pd.concat(all_pitchers, ignore_index=True)
        _write_atomic(combined, LIVE_DIR / "pitchers_statcast.csv")
        pitcher_rows = len(combined)
        log.info("Wrote %d pitcher rows to pitchers_statcast.csv", pitcher_rows)

    if all_batters:
        combined = pd.concat(all_batters, ignore_index=True)
        _write_atomic(combined, LIVE_DIR / "batters_statcast.csv")
        batter_rows = len(combined)
        log.info("Wrote %d batter rows to batters_statcast.csv", batter_rows)

    return {
        "statcast_pitcher_rows": pitcher_rows,
        "statcast_batter_rows":  batter_rows,
        "errors":                errors,
    }
