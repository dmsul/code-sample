import pandas as pd

from econtools import load_or_build

from epa_airpoll import (monitors_annual_summary, monitors_data,
                         valid_flag_panel, naaqs_assessment_monitors,
                         nonattainment_block_panel,
                         )

from util import pmrule_imp_year, MONITOR_MAX_YEAR
from util.env import data_path
from analysis.basic_data import monitors_block


CONSTANT_RANGE_DIFF = 2


def prep_monitor_analysis(df: pd.DataFrame, rule='pm25_12') -> pd.DataFrame:

    df = _merge_nonattainment_status(df, rule)

    df = _merge_regulatory_use_flag(df, rule)

    df = df[~(df['monitor_id'] == '06031_4_881011')]

    # Monitor reading from rule implementation year
    df = df.set_index('monitor_id')
    imp_year = pmrule_imp_year[rule]
    mean_in_year = df.loc[df['year'] == imp_year, 'arithmetic_mean']
    df = df.join(mean_in_year.to_frame('imp_year_mean'))
    naaqs_limit = _naaqs_limit(rule)

    df['post'] = df['year'] > imp_year
    df['nonattain_post'] = ((df['nonattain']) & (df['post'])).astype(int)

    df['over_naaqs'] = (df['imp_year_mean'] >= naaqs_limit)
    df['under_naaqs'] = (df['imp_year_mean'] < naaqs_limit)

    df['nonattain_over'] = (df['nonattain']) & (df['over_naaqs'])
    df['nonattain_under'] = (df['nonattain']) & (df['under_naaqs'])

    df['targeted'] = (df['used_for_naaqs']) & (df['nonattain_over'])
    df['untargeted'] = (df['nonattain']) & (~df['targeted'])

    df['targeted_post'] = (df['targeted']) & (df['post'])
    df['untargeted_post'] = (df['untargeted']) & (df['post'])

    df = df.reset_index()

    # Add year dummies
    for idx, year in enumerate(df['year'].unique()):
        if idx == 0:
            continue
        df[f'_Iyear_{year}'] = df['year'] == year

    return df

def _merge_nonattainment_status(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    mons_block = monitors_block()
    imp_year = pmrule_imp_year[rule]
    nonattain = nonattainment_block_panel(rule)[imp_year]
    mons_block = mons_block.to_frame('block_id').join(
        nonattain.to_frame('nonattain'), on='block_id')
    df = df.join(mons_block['nonattain'], on='monitor_id')
    df['nonattain'] = df['nonattain'].fillna(False)

    return df

def _merge_regulatory_use_flag(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    test_year = pmrule_imp_year[rule] - 1
    naaqs = valid_naaqs_monitors(test_year)
    df['used_for_naaqs'] = (df['monitor_id']
                            .isin(naaqs['monitor_id'].unique()))

    return df

def _naaqs_limit(rule):
    return 12 if rule == 'pm25_12' else 15

    return df


def constant_monitor_panel(rule: str='pm25_12') -> pd.DataFrame:
    df = monitors_summary_panel()

    # Define sample period
    imp_year = pmrule_imp_year[rule]
    constant_range_diff = CONSTANT_RANGE_DIFF
    year0 = imp_year - constant_range_diff
    yearT = min(imp_year + constant_range_diff, MONITOR_MAX_YEAR)

    df = df[(df['year'] >= year0) & (df['year'] <= yearT)]

    df = _restrict_to_continuous_in_sample(df, year0, yearT)

    return df

def semi_constant_monitor_panel(rule: str='pm25_12') -> pd.DataFrame:
    """
    Same as `constant_monitor_panel` but includes a few more years before the
    "constant" period where values may be missing.
    """
    df = monitors_summary_panel()

    # Define sample period
    imp_year = pmrule_imp_year[rule]
    constant_range_diff = CONSTANT_RANGE_DIFF
    sample_back_diff = 5
    year0 = imp_year - constant_range_diff
    yearT = min(imp_year + constant_range_diff, MONITOR_MAX_YEAR)
    year_T = imp_year - sample_back_diff
    df = df[(df['year'] >= year_T) & (df['year'] <= yearT)]

    df = _restrict_to_continuous_in_sample(df, year0, yearT)

    return df

def _restrict_to_continuous_in_sample(df: pd.DataFrame,
                                      year0: int,
                                      yearT: int
                                      ) -> pd.DataFrame:
    year_count = (df[(df['year'] >= year0) & (df['year'] <= yearT)]
                  .groupby('monitor_id')['arithmetic_mean']
                  .count())
    df = df.join(year_count.to_frame('year_count'), on='monitor_id')
    df = df[(df['year_count'] >= ((yearT - year0) + 1))]
    del df['year_count']

    return df


def valid_naaqs_monitors(year: int,
                         lag3: bool=True) -> pd.DataFrame:
    df = monitors_data()
    df = df[df['parameter_code'] == 88101]
    if lag3:
        valid_in_year = (valid_flag_panel()
                         .loc[:, year-3:year-1]
                         .min(axis=1)
                         .to_frame('is_valid'))
    else:
        valid_in_year = (valid_flag_panel()
                         .loc[:, year]
                         .to_frame('is_valid'))
    df = df.join(valid_in_year, on='monitor_id')
    df = df[df['is_valid'].fillna(False)]

    df['fips'] = (df['state_code'].astype(str).str.zfill(2) +
                  df['county_code'].astype(str).str.zfill(3))

    # Drop monitors as appropriate
    naaqs = naaqs_assessment_monitors()
    naaqs['in_list'] = True
    df['site_id'] = df['site_id'].apply(_fix_site_id)
    df = df.join(naaqs.set_index('fips')['in_list'], on='fips')
    wtf = ~((df['in_list']) &
            (~df['site_id'].isin(naaqs['site_id'].tolist())))

    is_primary = df['naaqs_primary_monitor'] == 'Y'
    df = df[(is_primary) & (wtf)]

    df = df.drop(['in_list', 'is_valid', 'start_date', 'end_date'], axis=1)

    return df

def _fix_site_id(x: str) -> str:
    a, b = x.split('_')
    return a + b.zfill(4)


@load_or_build(data_path('tmp_monitor_summ_panel.pkl'))
def monitors_summary_panel() -> pd.DataFrame:
    dfs = [monitors_summary_clean(year)
           for year in range(2000, MONITOR_MAX_YEAR + 1)]

    df = pd.concat(dfs)
    del dfs

    df = df[df['parameter_code'] == 88101]

    df['fips'] = (df['state_code'].astype(str).str.zfill(2) +
                  df['county_code'].astype(str).str.zfill(3))

    df = df.drop(['parameter_code', 'datum', 'parameter_name',
                  'site_number', 'poc', 'units_of_measure',
                  'state_code', 'county_code',
                  'latitude', 'longitude',
                  'arithmetic_standard_dev',
                  '1st_max_value', '1st_max_datetime',
                  '2nd_max_value', '2nd_max_datetime',
                  '3rd_max_value', '3rd_max_datetime',
                  '4th_max_value', '4th_max_datetime',
                  '1st_max_non_overlapping_value',
                  '1st_no_max_datetime',
                  '2nd_max_non_overlapping_value',
                  '2nd_no_max_datetime',
                  '99th_percentile',
                  '95th_percentile',
                  '90th_percentile',
                  '50th_percentile',
                  '10th_percentile',
                  'local_site_name',
                  'address',
                  'state_name',
                  'county_name',
                  'city_name',
                  'cbsa_name',
                  'date_of_last_change'],
                 axis=1)

    df = df.rename(columns={'latitude': 'y',
                            'longitude': 'x'})

    df = df[df['event_type'].isin(('No Events', 'Concurred Events Excluded'))]
    df = df[df['pollutant_standard'] == 'PM25 Annual 2006']

    assert df.shape == df.drop_duplicates(['year', 'monitor_id']).shape

    return df


def monitors_summary_clean(year: int) -> pd.DataFrame:
    df = monitors_annual_summary(year)

    try:
        df = df[df['state_code'] != 'CC']
    except TypeError:
        pass

    return df


if __name__ == "__main__":
    df = valid_naaqs_monitors(2015)
