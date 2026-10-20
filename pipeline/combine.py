"""
combine.py — Left-join pitcher_injuries onto pitchers_statcast on player_name.
Uses rapidfuzz fuzzy matching (threshold ≥ 90) to bridge name variations across
sources. Computes is_currently_injured and days_since_il. Writes atomic CSV.
"""
from __future__ import annotations

import datetime
import logging
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
LIVE_DIR = ROOT / "data" / "live"

PITCHER_CSV = LIVE_DIR / "pitchers_statcast.csv"
INJURY_CSV  = LIVE_DIR / "pitcher_injuries.csv"
OUTPUT_CSV  = LIVE_DIR / "pitchers_statcast.csv"   # overwrites in-place atomically

FUZZY_THRESHOLD = 90


def _deaccent(text: str) -> str:
    """Fold accented characters to their ASCII equivalents for name matching.

    'Carlos Rodón' → 'carlos rodon', bridging FanGraphs (ASCII) vs MLB API (UTF-8).
    """
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii").lower()


def _write_atomic(df: pd.DataFrame, target: Path) -> None:
    """Write to .tmp then rename — safe against partial-write corruption."""
    df = df.sort_values(
        ["season", "player_name"], ascending=[False, True]
    ).reset_index(drop=True)
    tmp = target.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.rename(target)


def _fuzzy_match(name: str, candidates: list[str]) -> str | None:
    """Return the best fuzzy match above FUZZY_THRESHOLD, or None."""
    try:
        from rapidfuzz import fuzz, process  # type: ignore[import-untyped]
        result = process.extractOne(
            name, candidates,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_THRESHOLD,
        )
        # rapidfuzz ≥ 3.0: returns (match, score, index) or None
        return result[0] if result else None
    except Exception as exc:
        log.warning("rapidfuzz error on '%s': %s", name, exc)
        return None


def _build_injury_index(injuries: pd.DataFrame) -> dict[str, pd.Series]:
    """Build a lookup dict from deaccented player_name → injury row.

    Deaccenting ('Rodón' → 'rodon') bridges the gap between FanGraphs (pure ASCII)
    and the MLB Stats API (UTF-8 with diacritics).
    """
    return {
        _deaccent(row["player_name"]): row
        for _, row in injuries.iterrows()
    }


def _is_currently_injured(row: pd.Series) -> bool:
    """True if injury_status is set and return_estimate is not yet past."""
    status = row.get("injury_status")
    if not status or pd.isna(status):
        return False
    ret = row.get("return_estimate", "TBD")
    if not ret or pd.isna(ret) or str(ret).strip().upper() == "TBD":
        return True   # on IL, no return date known — assume still injured
    try:
        return datetime.date.fromisoformat(str(ret)) >= datetime.date.today()
    except ValueError:
        return True   # unparseable date → assume still out


def _days_since_il(row: pd.Series) -> int | None:
    """Days since il_date, or None if il_date is absent or unparseable."""
    il = row.get("il_date")
    if not il or pd.isna(il):
        return None
    try:
        delta = datetime.date.today() - datetime.date.fromisoformat(str(il))
        return delta.days
    except ValueError:
        return None


def _join_injuries(
    pitchers: pd.DataFrame,
    injuries: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Left-join injuries onto pitchers. Exact match first; fuzzy fallback for misses.
    Returns (joined_df, list_of_unmatched_injury_names).
    """
    inj_index      = _build_injury_index(injuries)
    pitcher_names  = [n.strip().lower() for n in pitchers["player_name"].tolist()]
    inj_candidates = list(inj_index.keys())

    # Injury columns to merge in (excluding player_name to avoid conflict)
    inj_cols = [c for c in injuries.columns if c != "player_name"]

    matched_rows: list[dict] = []
    for _, p_row in pitchers.iterrows():
        # Deaccent the pitcher name (FanGraphs/BRef are ASCII; MLB API uses UTF-8 diacritics)
        p_key   = _deaccent(p_row["player_name"])
        # dict.get() already returns None on a miss; avoid `or None` which would try
        # to evaluate the truthiness of a pd.Series (ambiguous → ValueError).
        inj_row = inj_index.get(p_key)

        if inj_row is None:
            # Exact miss — try fuzzy (candidates are already deaccented keys)
            fuzzy_key = _fuzzy_match(p_key, inj_candidates)
            if fuzzy_key:
                inj_row = inj_index[fuzzy_key]
                log.debug("Fuzzy matched '%s' → '%s'", p_key, fuzzy_key)

        merged = p_row.to_dict()
        if inj_row is not None:
            for col in inj_cols:
                merged[col] = inj_row.get(col)
        else:
            for col in inj_cols:
                merged[col] = None

        merged["is_currently_injured"] = _is_currently_injured(
            inj_row if inj_row is not None else pd.Series(dtype=object)
        )
        merged["days_since_il"] = _days_since_il(
            inj_row if inj_row is not None else pd.Series(dtype=object)
        )
        matched_rows.append(merged)

    # Find injury records that matched no pitcher (log, do not drop silently)
    matched_inj_keys = set()
    for _, p_row in pitchers.iterrows():
        p_key = _deaccent(p_row["player_name"])
        if p_key in inj_index:
            matched_inj_keys.add(p_key)
        else:
            fk = _fuzzy_match(p_key, inj_candidates)
            if fk:
                matched_inj_keys.add(fk)

    unmatched = [
        injuries.loc[
            injuries["player_name"].map(_deaccent) == k, "player_name"
        ].iloc[0]
        for k in inj_index
        if k not in matched_inj_keys
    ]

    return pd.DataFrame(matched_rows), unmatched


def run() -> dict[str, Any]:
    """Load, join, enrich, and write pitchers_statcast.csv. Returns log stats."""
    errors:    list[str] = []
    unmatched: list[str] = []

    if not PITCHER_CSV.exists():
        msg = f"{PITCHER_CSV} not found — run fetch_statcast first"
        log.error(msg)
        return {"combined_rows": 0, "unmatched_injury_names": [], "errors": [msg]}

    pitchers = pd.read_csv(PITCHER_CSV, encoding="utf-8-sig")
    log.info("Loaded %d pitcher rows from CSV", len(pitchers))

    if INJURY_CSV.exists():
        injuries  = pd.read_csv(INJURY_CSV, encoding="utf-8-sig")
        log.info("Loaded %d injury rows from CSV", len(injuries))
        combined, unmatched = _join_injuries(pitchers, injuries)
        if unmatched:
            log.warning("Injury records unmatched to any pitcher (%d): %s", len(unmatched), unmatched)
    else:
        log.warning("%s not found — combine will skip injury join", INJURY_CSV)
        combined = pitchers.copy()
        combined["is_currently_injured"] = False
        combined["days_since_il"]        = None

    _write_atomic(combined, OUTPUT_CSV)
    log.info("Wrote %d combined rows to %s", len(combined), OUTPUT_CSV.name)

    return {
        "combined_rows":          len(combined),
        "unmatched_injury_names": unmatched,
        "errors":                 errors,
    }
