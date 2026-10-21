"""Offline tests for the DuckDB + dbt layer and the regression tool.

The real source CSVs and the live APIs aren't available in CI, so these run on
small fixture files (tests/fixtures/) and mocked pybaseball calls.
Run: python -m pytest tests/
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mlb")
    db = tmp / "mlb.duckdb"
    import migrate_to_duckdb as m
    mp = pytest.MonkeyPatch()
    mp.setattr(m, "DB_PATH", db)
    mp.setattr(m, "PITCHERS_CSV", FIX / "Pitchers_Stats.csv")
    mp.setattr(m, "BATTERS_CSV", FIX / "Batters_Stats.csv")
    assert m.migrate_data() == 0
    mp.undo()
    dbt = shutil.which("dbt") or str(Path(sys.executable).parent / "dbt")
    env = {**os.environ, "DBT_DB_PATH": str(db)}
    proj = tmp / "dbt"
    shutil.copytree(ROOT / "dbt", proj, ignore=shutil.ignore_patterns("target", "logs", ".user.yml"))
    run = subprocess.run([dbt, "build", "--project-dir", str(proj), "--profiles-dir", str(proj)],
                         env=env, capture_output=True, text=True, cwd=tmp)
    assert run.returncode == 0, run.stdout[-3000:]
    return db


def test_migration_fails_loudly_when_csvs_are_missing(tmp_path, monkeypatch):
    import migrate_to_duckdb as m
    monkeypatch.setattr(m, "PITCHERS_CSV", tmp_path / "nope.csv")
    assert m.migrate_data() == 1
    r = subprocess.run([sys.executable, str(ROOT / "migrate_to_duckdb.py")], cwd=tmp_path,
                       env={**os.environ, "DBT_DB_PATH": str(tmp_path / "x.duckdb")}, capture_output=True)
    if not (ROOT / "data" / "Pitchers_Stats.csv").exists():
        assert r.returncode == 1


def test_innings_and_fip(built):
    con = duckdb.connect(str(built), read_only=True)
    p = con.execute("SELECT player_id, innings_pitched, fip_proxy FROM raw_pitchers ORDER BY player_id").df()
    assert p.loc[2, "innings_pitched"] == pytest.approx(10 + 1 / 3)          # "10.1" = 10 1/3 IP
    assert p.loc[0, "fip_proxy"] == pytest.approx((13 * 20 + 3 * 50 - 2 * 200) / 180 + 3.20)


def test_era_regression_flag_compares_era_with_fip(built):
    con = duckdb.connect(str(built), read_only=True)
    f = dict(con.execute("SELECT player_id, era_regression_flag FROM mart_pitcher_profiles").fetchall())
    assert f[1001] == 1   # ERA 4.50 vs FIP 3.26: unlucky, should come down
    assert f[1002] == 0   # ERA 3.00 below FIP
    assert f[1003] == 0   # 13.50 ERA in 10 IP: too small to flag
    # The old rule (ERA > xwOBA + 0.50) flagged all three
    old = con.execute("SELECT COUNT(*) FROM mart_pitcher_profiles WHERE p_era > xwoba + 0.50").fetchone()[0]
    assert old == 3


def test_contact_quality_score_is_bounded_and_ordered(built):
    con = duckdb.connect(str(built), read_only=True)
    b = con.execute("SELECT player_id, contact_quality_score FROM stg_batters ORDER BY contact_quality_score").df()
    assert b["contact_quality_score"].between(0, 1).all()
    assert b["player_id"].tolist() == [2002, 2003, 2001]


def _fake_bref(n=80, seed=0):
    rng = np.random.default_rng(seed)
    ip_whole = rng.integers(31, 200, n)
    df = pd.DataFrame({
        "Name": [f"Pitcher {i}" for i in range(n)], "Tm": "NYY",
        "IP": ip_whole + rng.choice([0.0, 0.1, 0.2], n),
        "BF": ip_whole * 4.3, "SO": rng.integers(20, 220, n), "BB": rng.integers(5, 70, n),
        "HR": rng.integers(2, 30, n),
    })
    ip = df["IP"].floordiv(1) + (df["IP"].mod(1) * 10).round() / 3
    df["ERA"] = ((13 * df["HR"] + 3 * df["BB"] - 2 * df["SO"]) / ip + 3.2 + rng.normal(0, 0.6, n)).round(2)
    small = pd.DataFrame({"Name": ["Mop Up", "Opener"], "Tm": "NYY", "IP": [2.1, 12.0], "BF": [12, 50],
                          "SO": [1, 10], "BB": [3, 4], "HR": [2, 1], "ERA": [27.0, 4.5]})
    return pd.concat([df, small], ignore_index=True)


def test_regression_bref_fallback_filters_innings(monkeypatch):
    import pybaseball
    import agent_tools

    def fangraphs_403(*a, **k):
        raise RuntimeError("HTTP 403")
    monkeypatch.setattr(pybaseball, "pitching_stats", fangraphs_403)
    monkeypatch.setattr(pybaseball, "pitching_stats_bref", lambda season: _fake_bref())
    monkeypatch.setattr(pybaseball, "statcast_pitcher_exitvelo_barrels", fangraphs_403)
    monkeypatch.setattr(pybaseball.cache, "enable", lambda: None)
    out = json.loads(agent_tools.tool_build_regression(
        {"target_variable": "era", "feature_variables": ["k_pct", "bb_pct", "fip"], "season": 2025}))
    assert "N = 80 pitchers" in out["text"], out["text"]   # the 2 sub-30-IP pitchers are dropped
    r2 = float(out["text"].split("CV R²: ")[1].split()[0])
    assert r2 > 0.3                                        # ERA was built from FIP, so it should fit
