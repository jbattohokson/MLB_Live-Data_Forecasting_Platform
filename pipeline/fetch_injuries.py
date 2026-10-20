"""
fetch_injuries.py — Pitcher injury data from four sources in priority order:
  1. MLB Stats API (structured, most reliable)
  2. Rotowire injury-report page
  3. Yahoo Sports MLB injuries
  4. ESPN MLB injuries

Each source is independent. If a source is blocked, disallowed by robots.txt,
or raises any exception, log it and continue to the next. Never abort the run.
"""
from __future__ import annotations

import datetime
import logging
import re
import time
import urllib.robotparser
from pathlib import Path
from typing import Any

import pandas as pd
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import HTTPError, Timeout

log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent.parent
LIVE_DIR  = ROOT / "data" / "live"
CACHE_DIR = ROOT / "data" / "cache"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

INJURY_COLS = [
    "player_name", "name_suffix", "team", "position",
    "injury_type", "injury_status", "il_date", "return_estimate",
    "notes", "data_source", "pulled_at",
]

SUFFIX_RE = re.compile(r"\s+(Jr\.?|Sr\.?|II|III|IV|V)$", re.IGNORECASE)


# ── Shared utilities ──────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _normalize_player_name(raw: str) -> tuple[str, str]:
    """Strip suffix into a separate field; return (clean_name, suffix)."""
    raw = raw.strip()
    m   = SUFFIX_RE.search(raw)
    if m:
        return raw[: m.start()].strip().title(), m.group(1)
    return raw.title(), ""


def _robots_allowed(base_url: str) -> bool:
    """Return True if our User-Agent is permitted to fetch base_url."""
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{base_url}/robots.txt")
    try:
        rp.read()
        return rp.can_fetch(HEADERS["User-Agent"], base_url)
    except Exception as exc:
        log.warning("Could not read robots.txt for %s: %s — skipping source", base_url, exc)
        return False


def _cache_path(source: str) -> Path:
    today = datetime.date.today().isoformat()
    return CACHE_DIR / f"injuries_{source}_{today}.html"


def _save_html_cache(source: str, html: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(source).write_text(html, encoding="utf-8")


def _load_html_cache(source: str) -> str | None:
    """Load today's cached HTML or the most recent prior cache."""
    today_path = _cache_path(source)
    if today_path.exists():
        return today_path.read_text(encoding="utf-8")
    candidates = sorted(CACHE_DIR.glob(f"injuries_{source}_*.html"))
    if candidates:
        return candidates[-1].read_text(encoding="utf-8")
    return None


def _get_html(url: str, source: str) -> str | None:
    """Fetch HTML with timeout + retry on 429. Falls back to cache on failure."""
    import requests

    def _retry_once() -> str | None:
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            return r.text
        except Exception:
            return None

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        html = resp.text
        _save_html_cache(source, html)
        return html
    except Timeout:
        log.warning("%s timed out — trying cache", source)
        return _load_html_cache(source)
    except ReqConnectionError as exc:
        log.warning("%s connection error: %s — trying cache", source, exc)
        return _load_html_cache(source)
    except HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            log.warning("%s rate-limited — waiting 60s then retrying once", source)
            time.sleep(60)
            return _retry_once() or _load_html_cache(source)
        log.warning("%s HTTP %s — trying cache", source, exc.response.status_code if exc.response else "?")
        return _load_html_cache(source)


def _write_atomic(df: pd.DataFrame, target: Path) -> None:
    tmp = target.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.rename(target)


# ── Source 1: MLB Stats API ───────────────────────────────────────────────────

def _fetch_pitcher_id_lookup(year: int) -> dict[int, str]:
    """Build {player_id: position_abbreviation} for all *pitchers* active in `year`.

    One call to statsapi's league-wide player roster endpoint, instead of the
    30 per-team calls used for transactions. Used to filter the MLB Stats API
    transaction feed (which has no position field) down to pitchers only —
    without this, every IL transaction league-wide (any position) passes the
    "injured list" text filter and gets treated as a pitcher injury.
    Returns {} on any failure so callers can fall back to unfiltered behavior.
    """
    try:
        import statsapi  # type: ignore[import-untyped]
        result = statsapi.get("sports_players", {"sportId": 1, "season": year})
        people = result.get("people", []) if isinstance(result, dict) else []
        lookup = {
            p["id"]: p.get("primaryPosition", {}).get("abbreviation", "P")
            for p in people
            if p.get("primaryPosition", {}).get("type") == "Pitcher" and "id" in p
        }
        log.info("Pitcher ID lookup built: %d pitchers for %d", len(lookup), year)
        return lookup
    except Exception as exc:
        log.warning(
            "Could not build pitcher ID lookup for %d (%s) — MLB API IL "
            "transactions will not be position-filtered, so non-pitchers may "
            "appear in pitcher_injuries.csv as unmatched rows.",
            year, exc,
        )
        return {}


def _parse_statsapi_transactions(
    raw: list[dict], now: str, pitcher_lookup: dict[int, str] | None = None,
) -> list[dict]:
    """Convert statsapi transaction dicts to our canonical schema.

    When `pitcher_lookup` is non-empty, transactions for players not in it
    (i.e. non-pitchers) are dropped — the MLB Stats API transaction payload
    itself carries no position field, so without this filter every league-wide
    IL move (any position) passes through as a "pitcher injury" row.
    """
    pitcher_lookup = pitcher_lookup or {}
    rows = []
    for txn in raw:
        desc = txn.get("description", "") or ""
        if "injured list" not in desc.lower() and " IL" not in desc:
            continue
        person    = txn.get("person") or {}
        person_id = person.get("id")
        if pitcher_lookup and person_id not in pitcher_lookup:
            continue
        raw_name = person.get("fullName", "")
        name, sfx = _normalize_player_name(raw_name)
        type_desc = txn.get("typeDesc", "")
        rows.append({
            "player_name":     name,
            "name_suffix":     sfx,
            "team":            (txn.get("toTeam") or {}).get("abbreviation", ""),
            "position":        pitcher_lookup.get(person_id, ""),
            "injury_type":     _extract_injury_type(desc),
            "injury_status":   _classify_il_status(type_desc or desc),
            "il_date":         txn.get("date") or None,
            "return_estimate": "TBD",
            "notes":           desc[:280],
            "data_source":     "mlb_api",
            "pulled_at":       now,
        })
    return rows



# All 30 MLB team IDs for the statsapi.get('transactions') per-team query.
# statsapi requires teamId (or playerId/date) — global queries without teamId raise ValueError.
MLB_TEAM_IDS: dict[str, int] = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KC":  118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD":  135, "SEA": 136, "SF":  137,
    "STL": 138, "TB":  139, "TEX": 140, "TOR": 141, "WSH": 120,
}

_STATSAPI_INTER_CALL_DELAY = 0.15   # seconds — polite rate limit between team queries


def _fetch_mlb_api(year: int) -> list[dict] | None:
    """Pull IL transactions from MLB Stats API (statsapi package).

    statsapi.get('transactions') requires a teamId; global queries raise ValueError.
    We loop all 30 team IDs and aggregate, sleeping briefly between calls to avoid
    flooding the MLB endpoint.  Falls back to mlbstatsapi on ImportError.
    """
    now   = _now_utc()
    start = f"{year}-01-01"
    end   = f"{year}-12-31"

    # Primary: MLB-StatsAPI (pip install MLB-StatsAPI; imports as statsapi)
    try:
        import statsapi  # type: ignore[import-untyped]
        pitcher_lookup = _fetch_pitcher_id_lookup(year)
        all_txns: list[dict] = []
        for abbr, team_id in MLB_TEAM_IDS.items():
            try:
                result = statsapi.get(
                    "transactions",
                    {"startDate": start, "endDate": end, "teamId": team_id},
                )
                txns = result.get("transactions", []) if isinstance(result, dict) else []
                all_txns.extend(txns)
                log.debug("Team %s (%d): %d transactions", abbr, team_id, len(txns))
            except Exception as exc:
                log.warning("statsapi team %s (%d) failed: %s — skipping", abbr, team_id, exc)
            time.sleep(_STATSAPI_INTER_CALL_DELAY)

        rows = _parse_statsapi_transactions(all_txns, now, pitcher_lookup)
        log.info(
            "MLB API (statsapi) returned %d raw transactions → %d pitcher IL rows "
            "for %d (non-pitcher IL moves filtered out via pitcher ID lookup)",
            len(all_txns), len(rows), year,
        )
        return rows or None
    except ImportError:
        log.warning("statsapi not installed — trying mlbstatsapi")
    except Exception as exc:
        log.error("statsapi loop %d failed unexpectedly: %s", year, exc)

    # Fallback: python-mlbstatsapi
    try:
        import mlbstatsapi  # type: ignore[import-untyped]
        mlb = mlbstatsapi.Mlb()
        try:
            raw = mlb.get_team_transactions(0, start_date=start, end_date=end)
        except Exception:
            raw = mlb.get_transactions(start_date=start, end_date=end)
        txn_list = raw if isinstance(raw, list) else getattr(raw, "transactions", [])
        rows = _parse_statsapi_transactions(
            [
                {
                    "description": getattr(t, "description", ""),
                    "person": {"fullName": getattr(getattr(t, "person", None), "full_name", "")},
                    "date": getattr(t, "date", ""),
                    "typeDesc": getattr(t, "typeDesc", ""),
                    "toTeam": {},
                }
                for t in txn_list
            ],
            now,
        )
        log.info("MLB API (mlbstatsapi) returned %d IL transactions for %d", len(rows), year)
        return rows or None
    except ImportError:
        log.error("Neither statsapi nor mlbstatsapi is installed")
    except Exception as exc:
        log.error("mlbstatsapi transactions %d failed: %s", year, exc)
    return None


def _classify_il_status(text: str) -> str:
    text_lower = text.lower()
    if "60" in text_lower:
        return "60-Day IL"
    if "15" in text_lower:
        return "15-Day IL"
    if "10" in text_lower:
        return "10-Day IL"
    if "day-to-day" in text_lower or "dtd" in text_lower:
        return "DTD"
    if "season" in text_lower:
        return "Out for season"
    return "10-Day IL"


def _extract_injury_type(description: str) -> str:
    """Pull injury keyword from a transaction description string."""
    keywords = [
        "UCL", "elbow", "forearm", "shoulder", "hamstring", "oblique",
        "back", "knee", "ankle", "wrist", "finger", "hip", "lat",
        "triceps", "biceps", "groin", "calf", "toe", "neck",
    ]
    desc_lower = description.lower()
    for kw in keywords:
        if kw.lower() in desc_lower:
            return kw.title()
    return "Undisclosed"


# ── Source 2: Rotowire ────────────────────────────────────────────────────────

def _fetch_rotowire() -> list[dict] | None:
    """Scrape Rotowire injury table. Pitchers only (SP/RP/CL positions)."""
    url = "https://www.rotowire.com/baseball/injury-report.php"
    if not _robots_allowed("https://www.rotowire.com"):
        log.info("Rotowire disallows scraping via robots.txt — skipping")
        return None

    html = _get_html(url, "rotowire")
    if not html:
        return None

    try:
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]
        soup = BeautifulSoup(html, "lxml")
        rows_out, now = [], _now_utc()

        for row in soup.select("table.injury-report tr[class*='injury']"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < 7:
                continue
            pos = cells[2].upper()
            if pos not in ("SP", "RP", "CL", "P"):
                continue
            name, sfx = _normalize_player_name(cells[0])
            rows_out.append({
                "player_name":    name,
                "name_suffix":    sfx,
                "team":           cells[1],
                "position":       pos,
                "injury_type":    cells[3][:100],
                "injury_status":  _classify_il_status(cells[4]),
                "il_date":        None,
                "return_estimate": cells[5] or "TBD",
                "notes":          cells[6][:280] if len(cells) > 6 else "",
                "data_source":    "rotowire",
                "pulled_at":      now,
            })

        log.info("Rotowire returned %d pitcher injury rows", len(rows_out))
        return rows_out or None
    except Exception as exc:
        log.error("Rotowire parse failed: %s", exc)
        return None


# ── Source 3: Yahoo Sports ────────────────────────────────────────────────────

def _fetch_yahoo() -> list[dict] | None:
    """Scrape Yahoo Sports MLB injury table as a fallback."""
    url = "https://sports.yahoo.com/mlb/injuries/"
    if not _robots_allowed("https://sports.yahoo.com"):
        log.info("Yahoo Sports disallows scraping via robots.txt — skipping")
        return None

    html = _get_html(url, "yahoo")
    if not html:
        return None

    try:
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]
        soup = BeautifulSoup(html, "lxml")
        rows_out, now = [], _now_utc()

        for row in soup.select("table.InjuryTable tr"):
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if len(cells) < 5 or cells[0].lower() in ("player", ""):
                continue
            pos = cells[1].upper()
            if not any(p in pos for p in ("SP", "RP", "CL", "P")):
                continue
            name, sfx = _normalize_player_name(cells[0])
            rows_out.append({
                "player_name":    name,
                "name_suffix":    sfx,
                "team":           cells[2] if len(cells) > 2 else "",
                "position":       pos,
                "injury_type":    cells[3][:100] if len(cells) > 3 else "Undisclosed",
                "injury_status":  _classify_il_status(cells[4] if len(cells) > 4 else ""),
                "il_date":        None,
                "return_estimate": cells[5] if len(cells) > 5 else "TBD",
                "notes":          cells[6][:280] if len(cells) > 6 else "",
                "data_source":    "yahoo",
                "pulled_at":      now,
            })

        log.info("Yahoo returned %d pitcher injury rows", len(rows_out))
        return rows_out or None
    except Exception as exc:
        log.error("Yahoo parse failed: %s", exc)
        return None


# ── Source 4: ESPN ────────────────────────────────────────────────────────────

def _fetch_espn() -> list[dict] | None:
    """Scrape ESPN MLB injury page as the final fallback."""
    url = "https://www.espn.com/mlb/injuries"
    if not _robots_allowed("https://www.espn.com"):
        log.info("ESPN disallows scraping via robots.txt — skipping")
        return None

    html = _get_html(url, "espn")
    if not html:
        return None

    try:
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]
        soup = BeautifulSoup(html, "lxml")
        rows_out, now = [], _now_utc()

        for row in soup.select("table.Table tr.Table__TR"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < 4:
                continue
            pos = cells[1].upper() if len(cells) > 1 else ""
            if not any(p in pos for p in ("SP", "RP", "CL", "P")):
                continue
            name, sfx = _normalize_player_name(cells[0])
            rows_out.append({
                "player_name":    name,
                "name_suffix":    sfx,
                "team":           "",
                "position":       pos,
                "injury_type":    cells[2][:100] if len(cells) > 2 else "Undisclosed",
                "injury_status":  _classify_il_status(cells[3] if len(cells) > 3 else ""),
                "il_date":        None,
                "return_estimate": cells[4] if len(cells) > 4 else "TBD",
                "notes":          cells[5][:280] if len(cells) > 5 else "",
                "data_source":    "espn",
                "pulled_at":      now,
            })

        log.info("ESPN returned %d pitcher injury rows", len(rows_out))
        return rows_out or None
    except Exception as exc:
        log.error("ESPN parse failed: %s", exc)
        return None


# ── Entrypoint ────────────────────────────────────────────────────────────────

def run(year: int | None = None) -> dict[str, Any]:
    """Try all four sources in priority order. Return log stats dict."""
    if year is None:
        year = datetime.date.today().year

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    sources = [
        ("mlb_api",  lambda: _fetch_mlb_api(year)),
        ("rotowire", _fetch_rotowire),
        ("yahoo",    _fetch_yahoo),
        ("espn",     _fetch_espn),
    ]

    all_rows: list[dict] = []
    errors:   list[str]  = []
    used_sources:  list[str] = []

    for name, fetcher in sources:
        try:
            rows = fetcher()
        except Exception as exc:
            log.error("Unexpected error from %s: %s", name, exc)
            errors.append(f"{name}: {exc}")
            continue

        if rows:
            all_rows.extend(rows)
            used_sources.append(name)
            log.info("Source '%s' contributed %d rows", name, len(rows))

    if not all_rows:
        log.error("All injury sources failed or returned no data")
        return {"injury_rows": 0, "errors": errors}

    df = pd.DataFrame(all_rows).reindex(columns=INJURY_COLS)

    if len(df) < 10:
        log.warning("Only %d injury rows — scraper may be blocked or season is inactive", len(df))

    # Deduplicate: same player from multiple sources → keep the one ranked highest.
    source_rank = {"mlb_api": 0, "rotowire": 1, "yahoo": 2, "espn": 3}
    df["_rank"] = df["data_source"].map(source_rank).fillna(99)
    df = (
        df.sort_values("_rank")
        .drop_duplicates("player_name")
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )

    tmp = (LIVE_DIR / "pitcher_injuries.csv").with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.rename(LIVE_DIR / "pitcher_injuries.csv")
    log.info("Wrote %d injury rows from sources: %s", len(df), used_sources)

    return {"injury_rows": len(df), "errors": errors}
