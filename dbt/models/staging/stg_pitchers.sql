with source as (
    select * from {{ source('raw', 'raw_pitchers') }}
),
transformed as (
    select
        player_id,
        year                as season,
        player_name,
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
        p_formatted_ip,
        innings_pitched,
        woba_minus_xwoba,
        fip_proxy,
        whiff_tier
    from source
)
select * from transformed
