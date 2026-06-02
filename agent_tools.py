# MLB Live Agent: agent_tools.py
# Data fetch tools, ChromaDB helpers, Plotly chart builders,
# tool schemas, and system prompt for the agentic loop.
#
# ARCHITECTURE
#   Each tool function takes args: dict and returns a JSON string:
#     {"text": "...", "chart_type": "...", "chart_data": {...}}
#   - "text" is what Claude sees as the tool result
#   - "chart_type" / "chart_data" are parsed by MLB_Live_Agent.py
#     to render Plotly figures; absent when no chart applies
#   Errors are returned inside the JSON "text" field — never raised —
#   so Claude can read the message, self-correct, and retry.
#
# PYTHON VERSION
#   Requires Python 3.13.x. chromadb and sentence-transformers ship
#   binary wheels only through CPython 3.13. In VS Code: Command
#   Palette → Python: Select Interpreter → /usr/local/bin/python3.13

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if sys.version_info >= (3, 14):
    print(
        "\nWARNING: Python 3.14+ detected. chromadb and sentence-transformers\n"
        "  distribute pre-built wheels only up to CPython 3.13. Binary\n"
        "  extension failures (ModuleNotFoundError / RuntimeError) are likely.\n"
        f"  Active interpreter: {sys.executable}\n"
        "  Recommended fix:\n"
        "    1. VS Code → Command Palette → 'Python: Select Interpreter'\n"
        "    2. Choose: /usr/local/bin/python3.13\n"
        "    3. python3.13 -m pip install -r requirements.txt\n",
        file=sys.stderr,
    )

# TYPE_CHECKING is False at runtime, these imports only run for Pylance/mypy.
# anthropic, chromadb, and plotly are imported lazily inside each function
# that needs them so the module loads without them installed.
if TYPE_CHECKING:
    import chromadb                    # type: ignore[import-untyped]
    import plotly.graph_objects as go  # type: ignore[import-untyped]

ROOT            = Path(__file__).resolve().parent
CHROMA_DIR      = ROOT / "chroma_db"
EMBED_MODEL     = "all-MiniLM-L6-v2"
COLLECTION_NAME = "pitcher_tendencies"
CLAUDE_MODEL    = "claude-sonnet-4-6"
MAX_ITERATIONS  = 8
TOP_K           = 5

MLB_TEAM_IDS: dict[str, int] = {
    "SF": 137, "LAD": 119, "SD": 135, "COL": 115, "ARI": 109,
    "ATL": 144, "NYM": 121, "PHI": 143, "MIA": 146, "WSH": 120,
    "CHC": 112, "MIL": 158, "STL": 138, "CIN": 113, "PIT": 134,
    "HOU": 117, "TEX": 140, "SEA": 136, "LAA": 108, "OAK": 133,
    "NYY": 147, "BOS": 111, "TOR": 141, "BAL": 110, "TB":  139,
    "CLE": 114, "MIN": 142, "CWS": 145, "DET": 116, "KC":  118,
}

PITCH_COLORS: dict[str, str] = {
    "FF": "#e63946", "SI": "#f4a261", "FC": "#e9c46a",
    "SL": "#2a9d8f", "CU": "#264653", "KC": "#457b9d",
    "CH": "#6d6875", "FS": "#a8dadc", "EP": "#95d5b2",
    "KN": "#b7e4c7", "ST": "#52b788", "SV": "#f72585",
}

OUTCOME_COLORS: dict[str, str] = {
    "called_strike":   "#e63946",
    "swinging_strike": "#f4a261",
    "ball":            "#2a9d8f",
    "hit_into_play":   "#264653",
    "foul":            "#457b9d",
    "blocked_ball":    "#6d6875",
}


# ChromaDB collection helper
# Handles the 1.5.x import path change for
# SentenceTransformerEmbeddingFunction. Returns None (rather than sys.exit)
# when require_populated=True and the store is empty, so Streamlit can display
# a friendly "ChromaDB empty" message instead of crashing.

def _get_chroma_collection(require_populated: bool = False):
    import chromadb  # type: ignore[import-untyped]
    try:
        from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (  # type: ignore[import-untyped]
            SentenceTransformerEmbeddingFunction,
        )
    except ImportError:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction  # type: ignore[import-untyped]

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    except Exception as exc:
        # PersistentClient can fail on a locked or corrupted SQLite file.
        # Return None so callers degrade gracefully instead of crashing the app.
        print(f"WARNING: ChromaDB client failed to initialize: {exc}", file=sys.stderr)
        return None

    try:
        # SentenceTransformerEmbeddingFunction triggers a ~90MB model download on
        # first call. Wrap so a network outage or missing cache returns None rather
        # than raising inside the Streamlit main thread or the agent loop.
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    except Exception as exc:
        print(f"WARNING: Embedding model load failed: {exc}", file=sys.stderr)
        return None

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    if require_populated and collection.count() == 0:
        return None
    return collection


def _build_pitcher_summary(
    pitcher_name: str,
    pitch_mix: dict[str, float],
    velo_by_type: dict[str, float],
    xwoba: float | None,
    barrel_rate: float | None,
) -> str:
    dominant = max(pitch_mix, key=pitch_mix.get) if pitch_mix else "unknown"
    lines = [
        f"{pitcher_name} pitcher tendency profile.",
        f"Pitch mix (%): {pitch_mix}.",
        f"Avg velocity by pitch type (mph): {velo_by_type}.",
        f"Dominant pitch: {dominant}.",
    ]
    if xwoba is not None:
        lines.append(f"xwOBA allowed: {xwoba:.3f}.")
    if barrel_rate is not None:
        lines.append(f"Barrel rate allowed: {barrel_rate:.1f}%.")
    return " ".join(lines)


def _embed_pitcher(
    collection: Any,
    mlbam_id: int,
    pitcher_name: str,
    summary_text: str,
    season: str,
    extra_meta: dict,
) -> None:
    meta = {"pitcher_name": pitcher_name, "mlbam_id": mlbam_id, "season": season}
    meta.update({k: str(v) for k, v in extra_meta.items()})
    collection.upsert(
        ids=[f"{mlbam_id}_{season}"],
        documents=[summary_text],
        metadatas=[meta],
    )


def _lookup_player(pitcher_name: str) -> tuple[int, str] | None:
    import pybaseball  # type: ignore[import-untyped]

    name = pitcher_name.strip()

    # Two common input formats arrive here:
    #   "Logan Webb"   — typed naturally by the user
    #   "Webb, Logan"  — Baseball Reference / CSV copy-paste style
    # The comma branch handles the second case. Without it, "Webb, Logan".split()
    # produces ["Webb,", "Logan"] and parts[0] = "Webb," which pybaseball cannot
    # fuzzy-match because of the trailing comma.
    if "," in name:
        last, first = [part.strip() for part in name.split(",", 1)]
    else:
        parts = name.split()
        if len(parts) < 2:
            # Single-token names can't be looked up without a first name.
            # Return None rather than passing an empty string — pybaseball raises on empty.
            return None
        # Convention: first token = first name, last token = last name.
        # Multi-part last names ("De La Rosa") only pass the final token, but
        # pybaseball's fuzzy=True handles these well in practice.
        first, last = parts[0], parts[-1]

    lookup = pybaseball.playerid_lookup(last, first, fuzzy=True)
    if lookup.empty or lookup["key_mlbam"].isna().all():
        return None

    # Drop rows where MLBAM ID is missing — retired players pre-Statcast era sometimes
    # have a Fangraphs ID but no MLBAM ID, making them useless for statcast_pitcher() calls.
    row       = lookup.dropna(subset=["key_mlbam"]).iloc[0]
    full_name = f"{row['name_first'].title()} {row['name_last'].title()}"
    return int(row["key_mlbam"]), full_name


# Tool 1: Pitcher Statcast data

def tool_get_pitcher_statcast(args: dict) -> str:
    pitcher_name = args.get("pitcher_name", "").strip()
    start_date   = args.get("start_date", "").strip()
    end_date     = args.get("end_date", "").strip()

    if not all([pitcher_name, start_date, end_date]):
        return json.dumps({"text": "ERROR: pitcher_name, start_date, and end_date are all required."})

    try:
        import pybaseball  # type: ignore[import-untyped]
        pybaseball.cache.enable()

        result = _lookup_player(pitcher_name)
        if result is None:
            return json.dumps({"text": f"Player '{pitcher_name}' not found. Check spelling or try last name first."})
        mlbam_id, canonical_name = result

        df = pybaseball.statcast_pitcher(start_date, end_date, player_id=mlbam_id)
        if df is None or df.empty:
            return json.dumps({"text": f"No Statcast pitch data for {canonical_name} from {start_date} to {end_date}."})

        # Pitch mix
        pitch_counts  = df["pitch_type"].dropna().value_counts()
        total_pitches = int(pitch_counts.sum())
        pitch_mix     = {k: round(v / total_pitches * 100, 1) for k, v in pitch_counts.items()}

        # Avg velocity by pitch type
        velo_by_type = (
            df.groupby("pitch_type")["release_speed"].mean()
            .dropna().round(1).to_dict()
        )

        # xwOBA allowed
        xwoba = None
        if "estimated_woba_using_speedangle" in df.columns:
            vals  = df["estimated_woba_using_speedangle"].dropna()
            xwoba = round(float(vals.mean()), 3) if not vals.empty else None

        # Barrel rate
        barrel_rate = None
        if "barrel" in df.columns:
            barrels     = df["barrel"].dropna()
            barrel_rate = round(float(barrels.mean()) * 100, 1) if not barrels.empty else None

        # Sweet spot % (8–32° launch angle)
        sweet_spot_pct = None
        if "launch_angle" in df.columns:
            la = df["launch_angle"].dropna()
            if not la.empty:
                sweet_spot_pct = round(float(((la >= 8) & (la <= 32)).sum() / len(la) * 100), 1)

        # Whiff rate
        whiff_rate = None
        if "description" in df.columns:
            swing_desc = {"swinging_strike", "foul", "hit_into_play", "foul_tip",
                          "swinging_strike_blocked", "foul_bunt", "missed_bunt"}
            whiff_desc = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}
            swings     = df[df["description"].isin(swing_desc)]
            whiffs     = df[df["description"].isin(whiff_desc)]
            whiff_rate = round(len(whiffs) / len(swings) * 100, 1) if len(swings) > 0 else None

        # Embed pitcher summary to ChromaDB
        collection = _get_chroma_collection()
        if collection is not None:
            season  = start_date[:4]
            summary = _build_pitcher_summary(canonical_name, pitch_mix, velo_by_type, xwoba, barrel_rate)
            _embed_pitcher(collection, mlbam_id, canonical_name, summary, season, {
                "dominant_pitch": max(pitch_mix, key=pitch_mix.get) if pitch_mix else "N/A",
            })

        text = (
            f"{canonical_name} — Statcast Profile\n"
            f"Period: {start_date} to {end_date}  |  MLBAM ID: {mlbam_id}\n"
            f"Total pitches analyzed: {total_pitches}\n\n"
            f"Pitch Mix (%): {pitch_mix}\n"
            f"Avg Velocity by Pitch (mph): {velo_by_type}\n"
            f"xwOBA Allowed: {xwoba}\n"
            f"Barrel Rate: {barrel_rate}%\n"
            f"Sweet Spot %: {sweet_spot_pct}%\n"
            f"Whiff Rate: {whiff_rate}%"
        )
        return json.dumps({
            "text":       text,
            "chart_type": "pitcher_statcast",
            "chart_data": {
                "pitcher_name": canonical_name,
                "pitch_mix":    pitch_mix,
                "velo_by_type": velo_by_type,
            },
        })
    except Exception as e:
        return json.dumps({"text": f"STATCAST ERROR: {e}"})


# Tool 2: IL stints

def tool_get_il_stints(args: dict) -> str:
    team_abbr = args.get("team_name", "").strip().upper()
    season    = int(args.get("season", 2025))

    if not team_abbr:
        return json.dumps({"text": "ERROR: team_name is required (e.g. 'LAD')."})
    if team_abbr not in MLB_TEAM_IDS:
        return json.dumps({"text": f"ERROR: Unknown team '{team_abbr}'. Valid options: {sorted(MLB_TEAM_IDS.keys())}"})

    try:
        import mlbstatsapi  # type: ignore[import-untyped]
        mlb       = mlbstatsapi.Mlb()
        team_id   = MLB_TEAM_IDS[team_abbr]
        start_str = f"{season}-01-01"
        end_str   = f"{season}-12-31"

        try:
            raw = mlb.get_team_transactions(team_id, start_date=start_str, end_date=end_str)
        except Exception:
            raw = mlb.get_transactions(team_id=team_id, start_date=start_str, end_date=end_str)

        stints = []
        for txn in (raw if isinstance(raw, list) else getattr(raw, "transactions", [])):
            desc = getattr(txn, "description", "") or ""
            if "injured list" not in desc.lower() and " IL" not in desc:
                continue
            player_obj  = getattr(txn, "person", None)
            player_name = (
                getattr(player_obj, "full_name", None)
                or getattr(player_obj, "fullName", "Unknown")
                if player_obj else "Unknown"
            )
            date_str   = getattr(txn, "date", "")
            type_label = getattr(txn, "type_desc", "") or getattr(txn, "typeDesc", "IL")
            stints.append({
                "player_name": player_name,
                "date":        date_str,
                "type":        type_label,
                "description": desc,
            })

        if not stints:
            return json.dumps({"text": f"No IL transactions found for {team_abbr} in {season}."})

        shown   = stints[:30]
        omitted = len(stints) - len(shown)

        lines = [f"{team_abbr} IL Transactions — {season}  ({len(stints)} total)\n"]
        for s in shown:
            lines.append(f"  {s['date']:<12} {s['player_name']:<25} {s['type']}")

        if omitted > 0:
            # Surface the truncation explicitly so Claude can acknowledge it in its
            # response rather than presenting a partial list as if it were complete.
            lines.append(f"\n  ... {omitted} additional transactions not shown (display capped at 30).")
            lines.append(f"  Use a narrower date range or filter by player to see all entries.")

        text = "\n".join(lines)

        return json.dumps({
            "text":       text,
            "chart_type": "il_timeline",
            "chart_data": {"team": team_abbr, "season": season, "stints": stints},
        })
    except Exception as e:
        return json.dumps({"text": f"IL STINTS ERROR: {e}"})


# Tool 3: Pitch velocity trend 

def tool_get_velocity_trend(args: dict) -> str:
    pitcher_name = args.get("pitcher_name", "").strip()
    season       = int(args.get("season", 2025))

    if not pitcher_name:
        return json.dumps({"text": "ERROR: pitcher_name is required."})

    try:
        import pybaseball  # type: ignore[import-untyped]
        pybaseball.cache.enable()

        result = _lookup_player(pitcher_name)
        if result is None:
            return json.dumps({"text": f"Player '{pitcher_name}' not found."})
        mlbam_id, canonical_name = result

        df = pybaseball.statcast_pitcher(f"{season}-03-01", f"{season}-11-30", player_id=mlbam_id)
        if df is None or df.empty:
            return json.dumps({"text": f"No data found for {canonical_name} in {season}."})

        if "game_date" not in df.columns or "release_speed" not in df.columns:
            return json.dumps({"text": "Required columns (game_date, release_speed) not in dataset."})

        df["game_date"] = pd.to_datetime(df["game_date"])
        keep_types      = ["FF", "SI", "FC"]
        df_fb           = df[df["pitch_type"].isin(keep_types)].copy()

        trend = (
            df_fb.groupby(["game_date", "pitch_type"])["release_speed"]
            .mean().round(1).reset_index()
        )
        trend["game_date"] = trend["game_date"].astype(str)
        records = trend.to_dict(orient="records")

        text = (
            f"{canonical_name} — Velocity Trend ({season})\n"
            f"Game appearances tracked: {df['game_date'].nunique()}\n"
            f"Pitch types shown: {keep_types}\n"
            f"Date range: {trend['game_date'].min()} to {trend['game_date'].max()}"
        )
        return json.dumps({
            "text":       text,
            "chart_type": "velocity_trend",
            "chart_data": {"pitcher_name": canonical_name, "season": season, "records": records},
        })
    except Exception as e:
        return json.dumps({"text": f"VELOCITY TREND ERROR: {e}"})


# Tool 4: Team schedule 

def tool_get_schedule(args: dict) -> str:
    team_abbr  = args.get("team_name", "SF").strip().upper()
    start_date = args.get("start_date", "").strip()
    end_date   = args.get("end_date", "").strip()

    if not all([team_abbr, start_date, end_date]):
        return json.dumps({"text": "ERROR: team_name, start_date, end_date are all required."})
    if team_abbr not in MLB_TEAM_IDS:
        return json.dumps({"text": f"Unknown team '{team_abbr}'."})

    try:
        import mlbstatsapi  # type: ignore[import-untyped]
        mlb     = mlbstatsapi.Mlb()
        team_id = MLB_TEAM_IDS[team_abbr]

        schedule = mlb.get_schedule(
            start_date=start_date, end_date=end_date,
            sport_id=1, team_id=team_id,
        )

        games = []
        for date_obj in (getattr(schedule, "dates", []) or []):
            for game in (getattr(date_obj, "games", []) or []):
                home_team = getattr(getattr(getattr(game, "teams", None), "home", None), "team", None)
                away_team = getattr(getattr(getattr(game, "teams", None), "away", None), "team", None)
                home_name = getattr(home_team, "name", "?") if home_team else "?"
                away_name = getattr(away_team, "name", "?") if away_team else "?"
                game_date = getattr(date_obj, "date", "?")
                is_home   = team_id == getattr(home_team, "id", -1) if home_team else False
                opponent  = away_name if is_home else home_name
                games.append({
                    "date":      game_date,
                    "opponent":  opponent,
                    "home_away": "HOME" if is_home else "AWAY",
                    "game_pk":   getattr(game, "game_pk", None),
                })

        if not games:
            return json.dumps({"text": f"No games found for {team_abbr} from {start_date} to {end_date}."})

        lines = [f"{team_abbr} Schedule: {start_date} to {end_date}  ({len(games)} games)\n"]
        for g in games:
            lines.append(f"  {g['date']:<12} {g['home_away']:<5} vs {g['opponent']}")
        return json.dumps({"text": "\n".join(lines)})
    except Exception as e:
        return json.dumps({"text": f"SCHEDULE ERROR: {e}"})


# Tool 5: ChromaDB semantic search 

def tool_search_pitcher_profile(args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"text": "ERROR: query is required."})

    try:
        collection = _get_chroma_collection(require_populated=True)
        if collection is None:
            return json.dumps({"text": "ChromaDB is empty. Ask about a specific pitcher first to populate the index."})

        results = collection.query(
            query_texts=[query],
            n_results=min(TOP_K, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        if not results["documents"] or not results["documents"][0]:
            return json.dumps({"text": "No matching pitcher profiles found."})

        docs  = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        sims  = [round(1 - d / 2, 4) for d in dists]

        chunks = []
        for doc, meta, sim in zip(docs, metas, sims):
            name = meta.get("pitcher_name", "Unknown")
            chunks.append(f"[{name} | Similarity: {sim:.3f}]\n{doc}")

        return json.dumps({"text": "\n\n---\n\n".join(chunks)})
    except Exception as e:
        return json.dumps({"text": f"SEARCH ERROR: {e}"})


# Tool 6: Regression model 

FEATURE_MAP: dict[str, str] = {
    "k_pct":              "K%",
    "bb_pct":             "BB%",
    "fip":                "FIP",
    "xfip":               "xFIP",
    "babip":              "BABIP",
    "lob_pct":            "LOB%",
    "barrel_pct":         "barrel_batted_rate",
    "hard_hit_percent":   "hard_hit_percent",
    "exit_velocity_avg":  "exit_velocity_avg",
    "sweet_spot_percent": "sweet_spot_percent",
    "whiff_percent":      "whiff_percent",
}

TARGET_MAP: dict[str, str] = {
    "era":   "ERA",
    "xwoba": "xwOBA",
}


def tool_build_regression(args: dict) -> str:
    target      = args.get("target_variable", "era").strip().lower()
    features    = [f.strip().lower() for f in args.get("feature_variables", ["k_pct", "bb_pct", "fip"])]
    season      = int(args.get("season", 2025))
    team_filter = args.get("team_name", "").strip() or None

    target_col = TARGET_MAP.get(target)
    if target_col is None:
        return json.dumps({"text": f"ERROR: target_variable must be 'era' or 'xwoba', got '{target}'."})

    try:
        import pybaseball  # type: ignore[import-untyped]
        from sklearn.linear_model import Ridge                   # type: ignore[import-untyped]
        from sklearn.preprocessing import StandardScaler         # type: ignore[import-untyped]
        from sklearn.model_selection import cross_val_score      # type: ignore[import-untyped]

        pybaseball.cache.enable()
        fg = pybaseball.pitching_stats(season, season, qual=30)

        # Attempt to merge Statcast exit-velo / barrel data
        try:
            sc = pybaseball.statcast_pitcher_exitvelo_barrels(season, minPA=50)
            fg["_name_key"] = fg["Name"].str.lower().str.strip()
            sc["_name_key"] = sc["player_name"].str.lower().str.strip()
            merged = fg.merge(sc, on="_name_key", how="left")
        except Exception:
            merged = fg.copy()

        if team_filter:
            merged = merged[merged["Team"].str.contains(team_filter, case=False, na=False)]

        # Resolve requested features to actual DataFrame columns
        feature_cols  = []
        missing_feats = []
        for f in features:
            col = FEATURE_MAP.get(f, f)
            if col in merged.columns:
                feature_cols.append(col)
            else:
                missing_feats.append(f)

        if missing_feats:
            return json.dumps({
                "text": (
                    f"ERROR: Feature(s) not found: {missing_feats}.\n"
                    f"Available: {sorted(FEATURE_MAP.keys())}"
                )
            })
        if not feature_cols:
            return json.dumps({"text": "ERROR: No valid features resolved."})
        if target_col not in merged.columns:
            return json.dumps({"text": f"ERROR: Target '{target_col}' not in dataset."})

        df_model = merged[["Name", *feature_cols, target_col]].dropna()

        # Ridge regression needs enough data per fold for stable coefficient estimates.
        # StandardScaler inside cross_val_score fits on the training split of each fold —
        # with fewer than ~10 training samples the scaler's mean/std are unreliable.
        # Threshold of 30 matches the minimum innings-pitched filter on the source data.
        MIN_SAMPLES_FOR_REGRESSION = 30
        if len(df_model) < MIN_SAMPLES_FOR_REGRESSION:
            return json.dumps({
                "text": (
                    f"Only {len(df_model)} pitchers remain after filtering and dropna. "
                    f"Ridge regression requires at least {MIN_SAMPLES_FOR_REGRESSION} samples "
                    f"for stable cross-validation. Try a different season or remove the team filter."
                )
            })

        X        = df_model[feature_cols].values.astype(float)
        y        = df_model[target_col].values.astype(float)
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model    = Ridge(alpha=1.0)
        model.fit(X_scaled, y)

        # Fold count: each fold gets at least 3 test samples.
        # min(5, ...) caps at standard 5-fold CV.
        # max(2, ...) prevents sklearn from raising on cv=1.
        n_folds = min(5, max(2, len(df_model) // 3))
        cv_r2   = float(
            cross_val_score(model, X_scaled, y, cv=n_folds, scoring="r2").mean()
        )
        preds = model.predict(X_scaled)

        # Warn Claude when fewer than 5 folds were used so it can relay the caveat.
        cv_note = (
            f"  Note: Only {n_folds}-fold CV used (n={len(df_model)} pitchers). "
            f"Interpret R² with caution.\n"
            if n_folds < 5 else ""
        )

        feature_importance = sorted(
            [{"feature": col, "coefficient": round(float(coef), 4)}
             for col, coef in zip(feature_cols, model.coef_)],
            key=lambda x: abs(x["coefficient"]),
            reverse=True,
        )
        predictions = [
            {"pitcher": name, "actual": round(float(a), 3), "predicted": round(float(p), 3)}
            for name, a, p in zip(df_model["Name"], y, preds)
        ]

        coef_lines = "\n".join(
            f"  {fi['feature']}: {fi['coefficient']:+.4f}" for fi in feature_importance
        )
        text = (
            f"Ridge Regression — {target_col} ~ {' + '.join(feature_cols)}\n"
            f"Season: {season}  |  N = {len(df_model)} pitchers (min 30 IP)\n"
            f"{cv_note}"
            f"{n_folds}-fold CV R²: {cv_r2:.3f}\n\n"
            f"Standardized Coefficients:\n{coef_lines}"
        )
        return json.dumps({
            "text":       text,
            "chart_type": "regression",
            "chart_data": {
                "predictions":        predictions,
                "target":             target_col,
                "feature_importance": feature_importance,
                "r2":                 round(cv_r2, 3),
            },
        })
    except Exception as e:
        return json.dumps({"text": f"REGRESSION ERROR: {e}"})


# Tool 7: Pitch heatmap / trajectory data 
def tool_get_pitch_heatmap(args: dict) -> str:
    pitcher_name = args.get("pitcher_name", "").strip()
    pitch_type   = args.get("pitch_type", "").strip().upper()
    start_date   = args.get("start_date", "").strip()
    end_date     = args.get("end_date", "").strip()

    if not all([pitcher_name, pitch_type, start_date, end_date]):
        return json.dumps({"text": "ERROR: pitcher_name, pitch_type, start_date, end_date are all required."})

    try:
        import pybaseball  # type: ignore[import-untyped]
        pybaseball.cache.enable()

        result = _lookup_player(pitcher_name)
        if result is None:
            return json.dumps({"text": f"Player '{pitcher_name}' not found."})
        mlbam_id, canonical_name = result

        df = pybaseball.statcast_pitcher(start_date, end_date, player_id=mlbam_id)
        if df is None or df.empty:
            return json.dumps({"text": f"No data for {canonical_name} from {start_date} to {end_date}."})

        df_pt = df[df["pitch_type"] == pitch_type].copy()
        if df_pt.empty:
            available = df["pitch_type"].dropna().unique().tolist()
            return json.dumps({"text": f"No '{pitch_type}' pitches found. Available: {available}"})

        # Location data
        df_loc  = df_pt[["plate_x", "plate_z"]].dropna()
        plate_x = df_loc["plate_x"].round(3).tolist()
        plate_z = df_loc["plate_z"].round(3).tolist()

        # Movement / trajectory data
        move_src = df_pt[[c for c in ["pfx_x", "pfx_z", "release_speed", "description"]
                          if c in df_pt.columns]]
        if "pfx_x" in move_src.columns and "pfx_z" in move_src.columns:
            move_src = move_src.dropna(subset=["pfx_x", "pfx_z"])
        pfx_x        = move_src["pfx_x"].round(3).tolist() if "pfx_x" in move_src.columns else []
        pfx_z        = move_src["pfx_z"].round(3).tolist() if "pfx_z" in move_src.columns else []
        speeds       = move_src["release_speed"].round(1).tolist() if "release_speed" in move_src.columns else []
        descriptions = move_src["description"].tolist() if "description" in move_src.columns else []

        n    = len(df_pt)
        text = (
            f"{canonical_name} — {pitch_type} Location Profile\n"
            f"Period: {start_date} to {end_date}  |  {n} pitches\n"
            + (
                f"Avg velocity: {df_pt['release_speed'].mean():.1f} mph\n"
                f"Avg H-break (pfx_x): {df_pt['pfx_x'].mean():.2f} in\n"
                f"Avg V-break (pfx_z): {df_pt['pfx_z'].mean():.2f} in"
                if "pfx_x" in df_pt.columns else ""
            )
        )
        return json.dumps({
            "text":       text,
            "chart_type": "pitch_heatmap",
            "chart_data": {
                "pitcher_name": canonical_name,
                "pitch_type":   pitch_type,
                "plate_x":      plate_x,
                "plate_z":      plate_z,
                "pfx_x":        pfx_x,
                "pfx_z":        pfx_z,
                "speeds":       speeds,
                "descriptions": descriptions,
            },
        })
    except Exception as e:
        return json.dumps({"text": f"HEATMAP ERROR: {e}"})


# Plotly chart builders 
# Each returns a go.Figure. Called by MLB_Live_Agent.py after parsing chart_data.
# Plotly is imported lazily inside each function.

def plot_pitch_type_distribution(pitch_mix: dict[str, float], pitcher_name: str):
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    pitch_types = list(pitch_mix.keys())
    pcts        = list(pitch_mix.values())
    colors      = [PITCH_COLORS.get(pt, "#888888") for pt in pitch_types]
    fig = go.Figure(go.Bar(
        x=pitch_types, y=pcts,
        marker_color=colors,
        text=[f"{p}%" for p in pcts],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{pitcher_name} — Pitch Mix",
        xaxis_title="Pitch Type", yaxis_title="Usage %",
        template="plotly_dark", margin=dict(t=50, b=40),
    )
    return fig


def plot_velocity_by_pitch_type(velo_by_type: dict[str, float], pitcher_name: str):
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    sorted_items = sorted(velo_by_type.items(), key=lambda x: x[1])
    pitches = [k for k, _ in sorted_items]
    velos   = [v for _, v in sorted_items]
    colors  = [PITCH_COLORS.get(pt, "#888888") for pt in pitches]
    fig = go.Figure(go.Bar(
        x=velos, y=pitches, orientation="h",
        marker_color=colors,
        text=[f"{v} mph" for v in velos],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{pitcher_name} — Avg Velocity by Pitch Type",
        xaxis_title="Avg Velocity (mph)",
        template="plotly_dark", margin=dict(t=50, l=60),
    )
    return fig


def plot_velocity_trend(records: list[dict], pitcher_name: str, season: int):
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    df  = pd.DataFrame(records)
    fig = go.Figure()
    for pt in df["pitch_type"].unique():
        sub = df[df["pitch_type"] == pt].sort_values("game_date")
        fig.add_trace(go.Scatter(
            x=sub["game_date"], y=sub["release_speed"],
            mode="lines+markers", name=pt,
            line=dict(color=PITCH_COLORS.get(pt, "#888888"), width=2),
        ))
    fig.update_layout(
        title=f"{pitcher_name} — Velocity Trend {season}",
        xaxis_title="Game Date", yaxis_title="Release Speed (mph)",
        template="plotly_dark", margin=dict(t=50),
    )
    return fig


def plot_pitch_heatmap(
    plate_x: list[float], plate_z: list[float],
    pitch_type: str, pitcher_name: str,
):
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    fig = go.Figure()
    fig.add_trace(go.Histogram2dContour(
        x=plate_x, y=plate_z,
        colorscale="RdYlGn_r",
        showscale=True, ncontours=15,
        name="Density",
    ))
    fig.add_trace(go.Scatter(
        x=plate_x, y=plate_z,
        mode="markers",
        marker=dict(size=3, color=PITCH_COLORS.get(pitch_type, "#888"), opacity=0.3),
        name="Pitches",
    ))
    # Standard strike zone rectangle
    fig.add_shape(type="rect",
        x0=-0.83, x1=0.83, y0=1.5, y1=3.5,
        line=dict(color="white", width=2, dash="dash"),
    )
    fig.update_layout(
        title=f"{pitcher_name} — {pitch_type} Location Heatmap",
        xaxis=dict(title="Horizontal Position (ft)", range=[-2.5, 2.5]),
        yaxis=dict(title="Height (ft)", range=[0, 5]),
        template="plotly_dark", margin=dict(t=50),
    )
    return fig


def plot_3d_pitch_movement(
    pfx_x: list[float], pfx_z: list[float],
    speeds: list[float], descriptions: list[str],
    pitch_type: str, pitcher_name: str,
):
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    color_map = [OUTCOME_COLORS.get(d, "#888888") for d in descriptions]
    y_vals    = speeds if speeds else [0.0] * len(pfx_x)
    fig = go.Figure(go.Scatter3d(
        x=pfx_x, y=y_vals, z=pfx_z,
        mode="markers",
        marker=dict(size=3, color=color_map, opacity=0.7),
        text=descriptions,
        hovertemplate=(
            "H-Break: %{x:.2f} in<br>"
            "Speed: %{y:.1f} mph<br>"
            "V-Break: %{z:.2f} in<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=f"{pitcher_name} — {pitch_type} 3D Movement Profile",
        scene=dict(
            xaxis_title="H-Break (in)",
            yaxis_title="Velocity (mph)",
            zaxis_title="V-Break (in)",
        ),
        template="plotly_dark", margin=dict(t=50),
    )
    return fig


def plot_regression_results(
    predictions: list[dict], target: str,
    feature_importance: list[dict], r2: float,
):
    import plotly.graph_objects as go          # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Actual vs Predicted", "Feature Coefficients"],
    )

    actuals    = [p["actual"]    for p in predictions]
    predicteds = [p["predicted"] for p in predictions]
    names      = [p["pitcher"]   for p in predictions]

    fig.add_trace(go.Scatter(
        x=actuals, y=predicteds,
        mode="markers",
        marker=dict(size=6, color="#2a9d8f", opacity=0.7),
        text=names,
        hovertemplate="%{text}<br>Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>",
        name="Pitchers",
    ), row=1, col=1)

    mn, mx = min(actuals + predicteds), max(actuals + predicteds)
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines", line=dict(color="white", dash="dash"),
        name="Perfect Fit",
    ), row=1, col=1)

    coef_features = [fi["feature"]     for fi in feature_importance]
    coef_values   = [fi["coefficient"] for fi in feature_importance]
    bar_colors    = ["#e63946" if c > 0 else "#2a9d8f" for c in coef_values]
    fig.add_trace(go.Bar(
        x=coef_values, y=coef_features, orientation="h",
        marker_color=bar_colors, name="Coefficients",
    ), row=1, col=2)

    fig.update_layout(
        title=f"Ridge Regression: {target}  (CV R²={r2:.3f})",
        template="plotly_dark", showlegend=False, margin=dict(t=70),
    )
    fig.update_xaxes(title_text=f"Actual {target}",    row=1, col=1)
    fig.update_yaxes(title_text=f"Predicted {target}", row=1, col=1)
    return fig


def plot_il_timeline(stints: list[dict], team: str, season: int):
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    if not stints:
        fig = go.Figure()
        fig.update_layout(title="No IL data", template="plotly_dark")
        return fig

    df = pd.DataFrame(stints)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["date"]], y=[row["player_name"]],
            mode="markers+text",
            marker=dict(size=12, color="#e63946"),
            text=[row.get("type", "IL")],
            textposition="middle right",
            name=row["player_name"],
        ))
    fig.update_layout(
        title=f"{team} IL Transactions — {season}",
        xaxis_title="Date",
        template="plotly_dark",
        showlegend=False,
        margin=dict(t=50, l=160),
    )
    return fig


# Tool registry 
# TOOLS maps each name → (function, Anthropic schema).
# TOOL_SCHEMAS is derived from TOOLS and passed directly to the Anthropic API.

TOOLS: dict[str, tuple[Any, dict]] = {
    "get_pitcher_statcast": (
        tool_get_pitcher_statcast,
        {
            "name":        "get_pitcher_statcast",
            "description": (
                "Pull live Statcast pitch data for a named pitcher from Baseball Savant. "
                "Returns pitch type distribution (%), average velocity per pitch type, "
                "xwOBA allowed, barrel rate, sweet spot %, and whiff rate. "
                "Also embeds a pitcher tendency summary into ChromaDB for later semantic search. "
                "Call this first whenever a user asks about a specific pitcher."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pitcher_name": {"type": "string", "description": "Full name, e.g. 'Logan Webb'"},
                    "start_date":   {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date":     {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["pitcher_name", "start_date", "end_date"],
            },
        },
    ),
    "get_il_stints": (
        tool_get_il_stints,
        {
            "name":        "get_il_stints",
            "description": (
                "Retrieve Injured List transactions for a given MLB team and season. "
                "Returns IL placements with player names, dates, and descriptions. "
                "Useful for identifying which opposing pitchers may be at reduced effectiveness. "
                "Follow up with get_pitch_velocity_trend to check post-injury velocity drops."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "team_name": {"type": "string", "description": "Team abbreviation, e.g. 'LAD'"},
                    "season":    {"type": "integer", "description": "Season year, e.g. 2025"},
                },
                "required": ["team_name", "season"],
            },
        },
    ),
    "get_pitch_velocity_trend": (
        tool_get_velocity_trend,
        {
            "name":        "get_pitch_velocity_trend",
            "description": (
                "Pull game-by-game fastball velocity trend for a pitcher across a full season. "
                "Returns a time series (FF, SI, FC) for detecting post-injury velocity drops "
                "or fatigue patterns. Use after get_il_stints for injury impact analysis."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pitcher_name": {"type": "string"},
                    "season":       {"type": "integer"},
                },
                "required": ["pitcher_name", "season"],
            },
        },
    ),
    "get_schedule": (
        tool_get_schedule,
        {
            "name":        "get_schedule",
            "description": (
                "Get the MLB schedule for a team over a date range. "
                "Returns game dates, opponents, and home/away status. "
                "Use to identify upcoming opponents before pulling pitcher data."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "team_name":  {"type": "string", "description": "Team abbreviation, e.g. 'SF'"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date":   {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["team_name", "start_date", "end_date"],
            },
        },
    ),
    "search_pitcher_profile": (
        tool_search_pitcher_profile,
        {
            "name":        "search_pitcher_profile",
            "description": (
                "Semantic search over stored pitcher tendency summaries in ChromaDB. "
                "Use for natural-language questions like 'which pitcher throws the most sliders?' "
                "or 'find pitchers with declining velocity.' "
                "ChromaDB is populated automatically by get_pitcher_statcast — "
                "call that tool first if the store is empty."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query about pitcher tendencies"},
                },
                "required": ["query"],
            },
        },
    ),
    "build_regression_model": (
        tool_build_regression,
        {
            "name":        "build_regression_model",
            "description": (
                "Build a Ridge regression model predicting ERA or xwOBA from Statcast/FanGraphs features. "
                "Returns standardized coefficients, 5-fold CV R², and actual vs predicted per pitcher. "
                "feature_variables must be from: k_pct, bb_pct, fip, xfip, babip, lob_pct, "
                "barrel_pct, hard_hit_percent, exit_velocity_avg, sweet_spot_percent, whiff_percent."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "target_variable": {
                        "type":        "string",
                        "enum":        ["era", "xwoba"],
                        "description": "Dependent variable to predict",
                    },
                    "feature_variables": {
                        "type":        "array",
                        "items":       {"type": "string"},
                        "description": "Predictor names (see description for valid values)",
                    },
                    "season": {"type": "integer", "description": "MLB season year"},
                    "team_name": {
                        "type":        "string",
                        "description": "Optional team filter abbreviation",
                    },
                },
                "required": ["target_variable", "feature_variables", "season"],
            },
        },
    ),
    "get_pitch_heatmap_data": (
        tool_get_pitch_heatmap,
        {
            "name":        "get_pitch_heatmap_data",
            "description": (
                "Get pitch location (plate_x, plate_z) and movement data (pfx_x, pfx_z) "
                "for a specific pitcher and pitch type. Renders a 2D location heat map and "
                "a 3D movement fingerprint plot. "
                "Call get_pitcher_statcast first to discover which pitch types are available."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pitcher_name": {"type": "string"},
                    "pitch_type": {
                        "type":        "string",
                        "description": "Pitch type code: FF, SL, CH, CU, SI, KC, FC, FS, ST",
                    },
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date":   {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["pitcher_name", "pitch_type", "start_date", "end_date"],
            },
        },
    ),
}

TOOL_SCHEMAS: list[dict] = [schema for _, schema in TOOLS.values()]


# System prompt

SYSTEM_PROMPT = """\
You are an MLB pitching intelligence agent for the {my_team} coaching staff.
Today's date: {today}  |  Analysis window: {start_date} to {end_date}
Opponent team selected: {opponent}  |  Season: {season}

Your job: help the coaching staff understand opposing pitchers' tendencies,
identify injury-driven weaknesses, and provide actionable contact-quality insights
for the offense. Every answer must cite specific numbers from tool outputs.

You have seven tools:
  get_pitcher_statcast       — live pitch mix, velocity, xwOBA, barrel rate, whiff rate
  get_il_stints              — IL placements (dates + injury descriptions) for a team
  get_pitch_velocity_trend   — game-by-game velocity trend (injury / fatigue detection)
  get_schedule               — upcoming games and opponents
  search_pitcher_profile     — semantic search over ChromaDB pitcher summaries
  build_regression_model     — Ridge regression predicting ERA or xwOBA
  get_pitch_heatmap_data     — pitch location + movement data for heat maps and 3D plots

Operating rules:
- Always call at least one tool before answering. Never answer from memory alone.
- When asked about a specific pitcher: call get_pitcher_statcast first.
  Follow with get_pitch_heatmap_data for location/movement charts.
- For injury analysis: get_il_stints → get_pitch_velocity_trend to detect velocity drops.
- For semantic questions ("find pitchers with heavy sliders"): search_pitcher_profile.
  If ChromaDB is empty, call get_pitcher_statcast on relevant pitchers first.
- For regression / predictive questions: build_regression_model.
- Cite specific numbers (velocities, percentages, xwOBA values) in every response.
- If a tool returns an error, report what failed and suggest the correct parameter format.
- xwOBA context: league avg ~.315; <.300 = excellent for pitcher; >.350 = hittable.
- Barrel rate context: <5% excellent; 5-8% average; >10% hittable.
- Whiff rate context: >30% elite; 20-30% average; <20% below average.
"""
