from typing import Callable

import numpy as np
import pandas as pd
from shapely.geometry import Point

from econtools import load_or_build

from epa_airpoll import (blocks_population, monitors_annual_summary,
                         load_blocks_shape_info, monitors_data,
                         load_block_shape)
from multisatpm import msat_northamer_1year, multisat_conus_year


from util.env import data_path
from analysis.geo_exposure import (multisatpm_exposure_block_conus,
                                   multisatpm_exposure_bg_conus,
                                   _xy_to_int_multisat, _int_to_xy_multisat)


xy_int = ['x_int', 'y_int']
xy = ['x', 'y']


# Block Level
@load_or_build(data_path('block_has_monitor_{}.pkl'), path_args=[0])
def block_has_monitor(year: int) -> pd.Series:
    """
    Bool Series 'this block shares a 0.01 grid cell with a monitor',
    index `block_id`
    """
    df = load_blocks_shape_info()
    df[xy_int] = _xy_to_int_multisat(df[xy])
    df = df.reset_index()   # otherwise `block_id` lost in merge

    mons = counties_monitor(year)
    mons[xy_int] = _xy_to_int_multisat(mons[xy])

    colname = f'{year}_mon_mean'
    df = df.merge(mons[xy_int + [colname]], on=xy_int, how='left')

    df['has_mon'] = df[colname].notnull()

    df = df.set_index('block_id')['has_mon'].squeeze()

    return df


@load_or_build(data_path('blocks_multisatpm_withpop_panel.pkl'))
def blocks_multisatpm_withpop_panel() -> pd.DataFrame:
    dfs = [blocks_multisatpm_withpop(y) for y in range(2000, 2016 + 1)]
    df = pd.concat(dfs, axis=1)
    del dfs

    return df


@load_or_build(data_path('blocks_multisatpm_withpop_{}.pkl'), path_args=[0])
def blocks_multisatpm_withpop(year):
    df = (multisatpm_exposure_block_conus(year)
          .squeeze()
          .to_frame(year))
    df = merge_blocks_pop(df)

    df = df[df['pop'] > 0].copy()

    del df['pop']

    return df


def merge_blocks_pop(df):
    """ Join blocks' population to `df` w/ block_id index """
    pop = blocks_population()
    pop.index.name = 'block_id'
    new_df = df.join(pop)
    assert new_df['pop'].notnull().min()
    return new_df


def merge_blocks_xy(df):
    blocks = load_blocks_shape_info()[xy]
    new_df = df.join(blocks)
    assert new_df['x'].notnull().min()
    return new_df


# Block Group Level
@load_or_build(data_path('bg_multisatpm_withpop_panel.pkl'))
def bg_multisatpm_withpop_panel():
    dfs = [bg_multisatpm_withpop(y) for y in range(2000, 2016 + 1)]
    df = pd.concat(dfs, axis=1)
    del dfs

    return df


@load_or_build(data_path('bg_multisatpm_withpop_{}.pkl'), path_args=[0])
def bg_multisatpm_withpop(year):
    df = (multisatpm_exposure_bg_conus(year)
          .squeeze()
          .to_frame(year))
    df = merge_bg_pop(df)

    df = df[df['pop'] > 0].copy()

    del df['pop']

    return df


def merge_bg_pop(df):
    """ Join block group population to `df` w/ bg_id index """
    pop = blocks_population().to_frame('pop')
    pop['bg_id'] = pop.index.astype(str).str[0:12]
    pop = pop.groupby('bg_id')['pop'].sum().to_frame('pop')
    new_df = df.join(pop)
    assert new_df['pop'].notnull().min()
    return new_df


def counties_monitor(year):
    mon = monitors_annual_summary(year)

    mon = mon[(mon['parameter_code'] == 88101) &
              (mon['pollutant_standard'] == 'PM25 Annual 2006') &
              (mon['event_type'] == 'No Events')]

    mon = mon.rename(columns={'latitude': 'y', 'longitude': 'x',
                              'arithmetic_mean': f'{year}_mon_mean'})

    mon['fips'] = (mon['state_code'].astype(str).str.zfill(2) +
                   mon['county_code'].astype(str).str.zfill(3))

    # Keep only max monitor in county
    mon = mon.sort_values(['fips', f'{year}_mon_mean'])
    mon = mon.drop_duplicates('fips', keep='last')

    return mon


@load_or_build(data_path('monitors_block.pkl'))
def monitors_block():
    df = monitors_data()
    df = df[df['state_code'] != 'CC']
    df['state_code'] = df['state_code'].astype(int)
    df = df[df['state_code'] < 80]      # No Mexico!
    df = df.set_index('monitor_id')
    assert df.index.is_unique

    df = df[df['parameter_code'] == 88101]
    df = df[df['longitude'].notnull()]
    out_df = pd.Series(index=df.index)
    out_df.name = 'block_id'
    out_df.loc[:] = 'x'*15
    for state_code, state_mon in df.groupby('state_code'):

        print(f"Loading shape {state_code}...", end='')
        state_shape = load_block_shape(str(state_code).zfill(2))
        state_shape = state_shape.set_index('GEOID10')
        state_shape.index.name = 'block_id'
        print("done!")

        for county_code, county_mon in state_mon.groupby('county_code'):
            print(f"Finding in county {county_code}")
            county_str = str(county_code).zfill(3)
            county_shape = state_shape[state_shape['COUNTYFP10'] == county_str]
            for monitor_id, monitor_row in county_mon.iterrows():
                this_point = Point(monitor_row['longitude'],
                                   monitor_row['latitude'])
                winner = county_shape.contains(this_point)
                count = winner.sum()
                if count < 1 or winner.empty:
                    winner = state_shape.contains(this_point)
                    count = winner.sum()
                    assert count == 1
                elif count == 1:
                    pass
                else:
                    raise AssertionError

                out_df.at[monitor_id] = winner[winner].index[0]

    return out_df[out_df != 'x'*15]


# Satellite Data - Block Level
def multisatpm_years_block(start_year, end_year):       # XXX DEPRECATE
    years = range(start_year, (end_year+1))
    exp_dfs = [(multisatpm_exposure_block_conus(year).squeeze().to_frame(year))
               for year in years]
    df = pd.concat(exp_dfs, axis=1)
    del exp_dfs

    return df


@load_or_build(data_path('tmp_multisatpm_3year_wlag_block.pkl'))
def prep_multisatpm_3year_wlag_block():
    df = blocks_multisatpm_withpop_panel()
    out = panel_to_3lag(df)

    return out


@load_or_build(data_path('tmp_multisatpm_3year_block.pkl'))
def prep_multisatpm_3year_block():
    df = blocks_multisatpm_withpop_panel()
    out = panel_to_3nolag(df)

    return out


# Satellite Data - Block Group Level
def multisatpm_years_bg(start_year, end_year):
    years = range(start_year, (end_year+1))
    exp_dfs = [(multisatpm_exposure_bg_conus(year).squeeze().to_frame(year))
               for year in years]
    df = pd.concat(exp_dfs, axis=1)
    del exp_dfs

    return df


@load_or_build(data_path('tmp_multisatpm_3year_wlag_bg.pkl'))
def prep_multisatpm_3year_wlag_bg():
    df = bg_multisatpm_withpop_panel()
    out = panel_to_3lag(df)

    return out


@load_or_build(data_path('tmp_multisatpm_3year_bg.pkl'))
def prep_multisatpm_3year_bg():
    df = bg_multisatpm_withpop_panel()
    out = panel_to_3nolag(df)

    return out


# multisatpm aux's
@load_or_build(data_path('tmp_multisatpm_3lag_{}.pkl'), path_args=[0])
def multisatpm_3lag_year(year: int) -> pd.Series:
    return multisatpm_3lag_panel()[year].squeeze()


@load_or_build(data_path('tmp_multisatpm_3lag_panel.pkl'))
def multisatpm_3lag_panel() -> pd.DataFrame:
    df = multisatpm_panel()
    out = panel_to_3lag(df)
    return out


def multisatpm_panel() -> pd.DataFrame:
    year_func = multisat_conus_year
    return _panel_guts(year_func)


# msatna (new) data
@load_or_build(data_path('tmp_msatna_bg_3lag_{}.pkl'), path_args=[0])
def msatna_bg_3lag_year(year: int) -> pd.Series:
    df = msatna_blocks_3lag_year(year).to_frame(year)
    df = merge_blocks_pop(df)
    df['bg_id'] = df.index.astype(str).str[0:12]

    # pop weights
    bgs_pop = df.groupby('bg_id')['pop'].sum()
    df = df.join(bgs_pop.to_frame('bgs_pop'), on='bg_id')
    df['weight'] = df['pop'] / df['bgs_pop']
    df['exp_weighted'] = df[year] * df['weight']

    df = df.groupby('bg_id')['exp_weighted'].sum()
    df.name = year

    return df


@load_or_build(data_path('tmp_msatna_blocks_3lag_{}.pkl'), path_args=[0])
def msatna_blocks_3lag_year(year: int) -> pd.Series:
    """ Convenience, just for the caching """
    return msatna_blocks_3lag_panel()[year]


@load_or_build(data_path('tmp_msatna_blocks_3lag_panel.pkl'))
def msatna_blocks_3lag_panel() -> pd.DataFrame:
    exp = msatna_3lag_panel()
    df = merge_blocks_msatna(exp)

    return df


@load_or_build(data_path('tmp_msatna_blocks_panel.pkl'))
def msatna_blocks_panel() -> pd.DataFrame:
    exp = msatna_panel()
    df = merge_blocks_msatna(exp)

    return df


def merge_blocks_msatna(exp0: pd.DataFrame) -> pd.DataFrame:
    # blocks
    df = load_blocks_shape_info()
    df[xy_int] = _xy_to_int_multisat(df[xy])

    # exposure
    exp = exp0.reset_index()
    exp[xy_int] = _xy_to_int_multisat(exp[xy])
    exp = exp.set_index(xy_int).drop(xy, axis=1)

    df = df.join(exp, on=xy_int)
    df = df.drop(xy_int + xy + ['area'], axis=1)
    df.columns = df.columns.astype(int)

    return df


@load_or_build(data_path('tmp_msatna_3lag_{}.pkl'), path_args=[0])
def msatna_3lag_year(year: int) -> pd.Series:
    return msatna_3lag_panel()[year]


@load_or_build(data_path('tmp_msatna_3lag_panel.pkl'))
def msatna_3lag_panel() -> pd.DataFrame:
    df = msatna_panel()
    out = panel_to_3lag(df)
    return out


def msatna_panel() -> pd.DataFrame:
    year_func = msat_northamer_1year
    return _panel_guts(year_func)


def _panel_guts(year_func: Callable) -> pd.DataFrame:
    years = range(2002, 2016 + 1)
    exp_dfs = [_prep_df_for_merge(year_func(year), year)
               for year in years]
    df = pd.concat(exp_dfs, axis=1)
    del exp_dfs

    df = df.reset_index()
    df[xy] = _int_to_xy_multisat(df[xy_int])
    df = (df
          .set_index(xy)
          .drop(xy_int, axis=1))

    df.columns = df.columns.astype(int)

    return df


def _prep_df_for_merge(df0: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df0.reset_index()
    df[xy_int] = _xy_to_int_multisat(df[xy])
    df = (df
          .drop(xy, axis=1)
          .set_index(xy_int)
          .squeeze()
          .to_frame(year))
    return df


# Aux functions
def panel_to_3lag(df: pd.DataFrame) -> pd.DataFrame:
    N, __ = df.shape
    years = df.columns[3:].values.tolist()
    years.append(max(years) + 1)
    out = pd.DataFrame(np.zeros((N, len(years))),
                       index=df.index,
                       columns=years)

    for y in years:
        out[y] = df.loc[:, y-3:y-1].mean(axis=1)

    return out


def panel_to_3nolag(df: pd.DataFrame) -> pd.DataFrame:
    N, __ = df.shape
    years = df.columns[2:].values.tolist()
    out = pd.DataFrame(np.zeros((N, len(years))),
                       index=df.index,
                       columns=years)

    for y in years:
        out[y] = df.loc[:, y-2:y].mean(axis=1)

    return out


if __name__ == '__main__':
    df = msatna_blocks_3lag_panel()
