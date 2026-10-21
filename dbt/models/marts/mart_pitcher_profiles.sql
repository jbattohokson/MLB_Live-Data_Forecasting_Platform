-- era_regression_flag: 1 when ERA runs at least 0.50 above the pitcher's FIP proxy, i.e.
-- results are worse than strikeouts/walks/home runs suggest and ERA is likely to come down.
-- AUDITED 2026-10-21: this compared p_era with xwoba + 0.50. xwOBA is a rate around 0.300,
-- so ERA (~4.00) beat it for essentially every pitcher and the flag was ~always 1.
-- Innings are required so a 2-inning reliever with a 13.50 ERA isn't flagged on noise.
with pitchers as (
    select * from {{ ref('stg_pitchers') }}
)
select
    *,
    p_era - fip_proxy as era_minus_fip,
    case
        when innings_pitched >= 20 and p_era - fip_proxy >= 0.50 then 1
        else 0
    end as era_regression_flag
from pitchers
