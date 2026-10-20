with pitchers as (
    select * from {{ ref('stg_pitchers') }}
)
select
    *,
    case
        when p_era > xwoba + 0.50 then 1
        else 0
    end as era_regression_flag
from pitchers
