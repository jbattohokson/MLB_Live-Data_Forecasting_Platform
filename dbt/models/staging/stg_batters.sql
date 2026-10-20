with source as (
    select * from {{ source('raw', 'raw_batters') }}
),
transformed as (
    select
        player_id,
        year                    as season,
        player_name,
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
        iso,
        norm_barrel,
        norm_sweet_spot,
        norm_exit_velo,
        contact_quality_score
    from source
)
select * from transformed
