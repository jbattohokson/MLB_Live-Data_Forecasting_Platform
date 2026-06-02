import logging
import duckdb
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT         = Path(__file__).resolve().parent
DB_PATH      = ROOT / "mlb_analytics.duckdb"
PITCHERS_CSV = ROOT / "Datasets" / "Pitchers_Stats.csv"
BATTERS_CSV  = ROOT / "Datasets" / "Batters_Stats.csv"


def migrate_data() -> None:
    for path in (PITCHERS_CSV, BATTERS_CSV):
        if not path.exists():
            logging.error("Source file not found: %s", path)
            return

    try:
        # duckdb.connect() accepts a Path object directly in DuckDB >= 0.9.x.
        con = duckdb.connect(DB_PATH)
    except duckdb.Error as exc:
        logging.error("Cannot open database %s: %s", DB_PATH, exc)
        return

    try:
        # pathlib produces forward-slash paths on macOS/Linux; DuckDB handles both
        # separators on Windows. No .replace() manipulation needed.
        con.execute(
            """
            CREATE OR REPLACE TABLE raw_pitchers AS
            WITH base AS (
                SELECT
                    CAST(player_id AS INTEGER)                                  AS player_id,
                    CAST(year      AS INTEGER)                                  AS year,
                    "last_name, first_name"                                     AS player_name,
                    p_era,
                    xwoba,
                    woba,
                    babip,
                    whiff_percent,
                    k_percent,
                    bb_percent,
                    barrel_batted_rate,
                    sweet_spot_percent,
                    exit_velocity_avg,
                    hard_hit_percent,
                    strikeout,
                    walk,
                    home_run,
                    CAST(p_formatted_ip AS DOUBLE)                              AS p_formatted_ip,
                    FLOOR(CAST(p_formatted_ip AS DOUBLE))
                        + (CAST(p_formatted_ip AS DOUBLE) % 1) * 10.0 / 3.0   AS innings_pitched
                FROM read_csv_auto(?)
            )
            SELECT
                *,
                woba - xwoba                                                    AS woba_minus_xwoba,
                (13.0 * home_run + 3.0 * walk - 2.0 * strikeout)
                    / NULLIF(innings_pitched, 0) + 3.20                         AS fip_proxy,
                CASE
                    WHEN whiff_percent >= 30 THEN 'Elite'
                    WHEN whiff_percent >= 24 THEN 'Average'
                    ELSE 'Below Average'
                END                                                             AS whiff_tier
            FROM base
            """,
            [str(PITCHERS_CSV)],
        )
        p_count = con.execute("SELECT COUNT(*) FROM raw_pitchers").fetchone()[0]
        logging.info("Pitchers loaded: %d (expected ~606)", p_count)

        con.execute(
            """
            CREATE OR REPLACE TABLE raw_batters AS
            WITH base AS (
                SELECT
                    CAST(player_id AS INTEGER)                                  AS player_id,
                    CAST(year      AS INTEGER)                                  AS year,
                    "last_name, first_name"                                     AS player_name,
                    batting_avg,
                    slg_percent,
                    on_base_percent,
                    on_base_plus_slg,
                    woba,
                    xwoba,
                    babip,
                    k_percent,
                    bb_percent,
                    barrel_batted_rate,
                    sweet_spot_percent,
                    exit_velocity_avg,
                    hard_hit_percent,
                    whiff_percent,
                    swing_percent,
                    home_run,
                    strikeout,
                    walk,
                    pa,
                    ab,
                    slg_percent - batting_avg                                   AS iso
                FROM read_csv_auto(?)
            ),
            -- Within-season normalization: PARTITION BY year so each batter is ranked
            -- against their season peers rather than across the full 2015-2025 dataset.
            -- This removes era inflation (league-average exit velocity has risen ~1.5 mph
            -- since Statcast tracking began) and makes contact_quality_score directly
            -- comparable year-over-year as a season-relative rank.
            normed AS (
                SELECT
                    *,
                    (barrel_batted_rate - MIN(barrel_batted_rate) OVER (PARTITION BY year))
                        / NULLIF(
                            MAX(barrel_batted_rate) OVER (PARTITION BY year)
                            - MIN(barrel_batted_rate) OVER (PARTITION BY year), 0
                        )                                                       AS norm_barrel,
                    (sweet_spot_percent - MIN(sweet_spot_percent) OVER (PARTITION BY year))
                        / NULLIF(
                            MAX(sweet_spot_percent) OVER (PARTITION BY year)
                            - MIN(sweet_spot_percent) OVER (PARTITION BY year), 0
                        )                                                       AS norm_sweet_spot,
                    (exit_velocity_avg - MIN(exit_velocity_avg) OVER (PARTITION BY year))
                        / NULLIF(
                            MAX(exit_velocity_avg) OVER (PARTITION BY year)
                            - MIN(exit_velocity_avg) OVER (PARTITION BY year), 0
                        )                                                       AS norm_exit_velo
                FROM base
            )
            SELECT
                *,
                (norm_barrel + norm_sweet_spot + norm_exit_velo) / 3.0         AS contact_quality_score
            FROM normed
            """,
            [str(BATTERS_CSV)],
        )
        b_count = con.execute("SELECT COUNT(*) FROM raw_batters").fetchone()[0]
        logging.info("Batters loaded:  %d (expected ~1520)", b_count)
        logging.info("Tables written: raw_pitchers, raw_batters")

    except duckdb.Error as exc:
        logging.error("Migration failed: %s", exc)
    finally:
        con.close()


if __name__ == "__main__":
    migrate_data()
