import pandas as pd

from econtools import load_or_build

from epa_airpoll import nonattainment_block_panel

from util import pmrule_imp_year
from util.env import data_path
from analysis.basic_data import (prep_multisatpm_3year_wlag_block,
                                 msatna_blocks_3lag_year,
                                 block_has_monitor, merge_blocks_pop)


def fips_misclass_flag(year: int, rule: str, data: str):
    df = blocks_misclass_flag(year, rule, data)
    df = df[~df['nonattain']]       # Do NOT want actual non-attains
    has_misclass = (df.groupby('fips')['is_over'].max()
                    .to_frame('fips_misclass'))
    return has_misclass


@load_or_build(data_path('tmp_blocks_misclass_df_{year}_{rule}_{data}.pkl'))
def blocks_misclass_flag(year: int, rule: str, data: str) -> pd.DataFrame:

    if data == 'multisatpm':
        df = prep_multisatpm_3year_wlag_block()[year].to_frame('exp')
    elif data == 'msatna':
        df = msatna_blocks_3lag_year(year).to_frame('exp')
    else:
        raise ValueError(f"{data} no good")

    df['fips'] = df.index.str[:5]

    # Merge in has_monitor
    df = df.join(block_has_monitor(year).to_frame('has_mon_block'))
    df = df.join(
        df.groupby('fips')['has_mon_block'].max() .to_frame('has_mon_fips'),
        on='fips')

    # Merge population
    df = merge_blocks_pop(df)
    fips_pop = df.groupby('fips')['pop'].sum().to_frame('fips_pop')
    df = df.join(fips_pop, on='fips')

    # Merge non-attainment
    df = merge_nonatt(df, rule)
    fips_nonatt = df.groupby('fips')['nonattain'].max()
    df = df.join(fips_nonatt.to_frame('has_nonattain'), on='fips')

    # Flag mis-classified
    naaqs = 12 if rule == 'pm25_12' else 15
    df['is_over'] = df['exp'] >= naaqs
    has_over = df.groupby('fips')['is_over'].max()
    df = df.join(has_over.to_frame('has_over'), on='fips')

    # XXX the fips is wrong
    # df['misclassed_fips'] = (df['has_over']) & (~df['has_nonattain'])
    df['misclassed_block'] = (df['is_over']) & (~df['nonattain'])

    return df


def merge_nonatt(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    nonatt = nonattainment_block_panel(rule)[pmrule_imp_year[rule]]
    df = df.join(nonatt.to_frame('nonattain'))
    df['nonattain'] = df['nonattain'].fillna(False)

    return df


if __name__ == '__main__':
    df = blocks_misclass_flag(2015, 'pm25_12', 'msatna')
