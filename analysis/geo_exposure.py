import numpy as np
import pandas as pd
import numexpr as ne

from econtools import load_or_build

from modis.util import annual_mean
from modis.clean.raw import load_modis_year
from multisatpm import multisat_conus_year, msat_northamer_conus_3year
from epa_airpoll import blocks_population
from epa_airpoll.util import name_to_fips_xwalk
from epa_airpoll.clean.census.shapefiles import (load_bg_shape,
                                                 load_blocks_shape_info)

from util.env import data_path


# modis/bg-level
@load_or_build(data_path('all_years_modis_exposure_bg.pkl'))
def modis_exposure_bg_conus():
    years = range(2001, 2016)
    conus_dfs = [(modis_exposure_bg_conus_year(year).squeeze().to_frame(year))
                 for year in years]
    df = pd.concat(conus_dfs, axis=1)

    del conus_dfs

    df = pd.DataFrame(df)

    return df


@load_or_build(data_path('modis_exposure_bg_conus_{}.pkl'), path_args=[0])
def modis_exposure_bg_conus_year(year):
    states = sorted(name_to_fips_xwalk.keys())
    modis_data = load_modis_year(year).reset_index()
    state_dfs = [state_modis_exposure_bg(
                 state, year, modis_data=modis_data) for state in states]
    df = pd.concat(state_dfs)
    del state_dfs

    df = pd.DataFrame(df)

    return df


@load_or_build(data_path('modis_exposure_bg_{}_{}.pkl'), path_args=[0, 1])
def state_modis_exposure_bg(state, year, modis_data=None):
    # modis data
    if modis_data is None:
        modis_data = load_modis_year(year).reset_index()

    # census data
    df = load_bg_shape(name_to_fips_xwalk[state])

    # bounding box
    bbox = df.bounds
    bound_columns = ['x0', 'y0', 'x1', 'y1']
    bbox.columns = bound_columns
    xy0 = ['x0', 'y0']
    xy1 = ['x1', 'y1']
    bbox_buffer = 0.05
    bbox[xy0] = bbox[xy0] - bbox_buffer
    bbox[xy1] = bbox[xy1] + bbox_buffer

    df = df.join(bbox, how='left')

    # condense modis data by state
    state_x0 = df['x0'].min()
    state_x1 = df['x1'].max()
    state_y0 = df['y0'].min()
    state_y1 = df['y1'].max()

    in_state_bounds = ((modis_data['x'] > state_x0) &
                       (modis_data['x'] < state_x1) &
                       (modis_data['y'] > state_y0) &
                       (modis_data['y'] < state_y1)
                       )
    state_modis = modis_data[in_state_bounds]

    state_modis_x = state_modis['x'].values
    state_modis_y = state_modis['y'].values

    # identify all modis points in each block group
    df = df.reset_index(drop=True)  # make index 0 to N
    mean_modis = np.zeros(len(df))
    for row in df[bound_columns].itertuples():
        i, x0, y0, x1, y1 = row     # Note the order here
        ne_equation = (f'(state_modis_x > {x0}) & (state_modis_x < {x1}) &'
                       f'(state_modis_y > {y0}) & (state_modis_y < {y1})')
        in_bounds = ne.evaluate(ne_equation)
        this_bg_modis = state_modis[in_bounds]
        mean_modis[i] = annual_mean(this_bg_modis)

    bg_id = (df['STATE'].astype(str).str.zfill(2) +
             df['COUNTY'].astype(str).str.zfill(3) +
             df['TRACT'].astype(str).str.zfill(6) +
             df['BLKGRP'].astype(str).str.zfill(1))
    out_df = pd.Series(mean_modis,
                       index=bg_id.values)

    return out_df


# multisatpm/block and block-group level
@load_or_build(data_path('multisatpm_exposure_bg_conus_{year}.pkl'))
def multisatpm_exposure_bg_conus(year):
    return _multisat_exposure_guts(year, geounit='bg')


@load_or_build(data_path('multisatpm_exposure_block_conus_{year}.pkl'))
def multisatpm_exposure_block_conus(year):
    return _multisat_exposure_guts(year, geounit='block')

def _multisat_exposure_guts(year, geounit='block'):
    multisatpm = multisat_conus_year(year).reset_index()
    multisatpm = multisatpm.rename(columns={0: 'exposure'})

    df = load_blocks_shape_info()
    df = df.reset_index()

    df = _coord_trans_and_merge(df, multisatpm)

    df = df[['block_id', 'exposure']]
    df = df.set_index('block_id')

    if geounit == 'bg':
        # weight exposure by population; converge on bg_id
        pop = blocks_population()
        df = df.join(pop, how='left')

        # block_id 15 digits; bg_id 12 digits
        df['bg_id'] = df.index.astype(str).str[0:12]

        bgs_pop = df.groupby('bg_id')['pop'].sum()
        df = df.join(bgs_pop.to_frame('bgs_pop'), on='bg_id')
        df['weight'] = df['pop'] / df['bgs_pop']
        df['exp_weighted'] = df['exposure'] * df['weight']

        df = df.groupby('bg_id')['exp_weighted'].sum().to_frame('exposure')

    elif geounit == 'block':
        pass

    else:
        raise ValueError(f'Invalid geounit: {geounit}')

    return df

def _coord_trans_and_merge(block_data, multisat_data):
    multisat_data['x_int'] = _xy_to_int_multisat(multisat_data['x'])
    multisat_data['y_int'] = _xy_to_int_multisat(multisat_data['y'])

    block_data['x_int'] = _xy_to_int_multisat(block_data['x'])
    block_data['y_int'] = _xy_to_int_multisat(block_data['y'])

    df = block_data.merge(multisat_data, on=['x_int', 'y_int'], how='left')

    return df

def _xy_to_int_multisat(x):
    return np.floor(x * 100).astype(int) * 10 + 5

def _int_to_xy_multisat(x):
    return x / 1000


# old msatna/block-level
@load_or_build(data_path('msat_v04NA01_exposure_block_conus_{year}.pkl'))
def msatna_v04NA01_exposure_block_conus(year):
    msat = (msat_northamer_conus_3year(year)
            .to_frame('exposure')
            .reset_index())

    df = load_blocks_shape_info()  # loads conus and does renaming
    df = df.reset_index()

    df = _coord_trans_and_merge_msat_v04NA01(df, msat)

    return df

def _coord_trans_and_merge_msat_v04NA01(block_data, multisat_data):
    multisat_data['x_int'] = _xy_to_int_msat_v04NA01(multisat_data['x'])
    multisat_data['y_int'] = _xy_to_int_msat_v04NA01(multisat_data['y'])

    block_data['x_int'] = _xy_to_int_msat_v04NA01(block_data['x'])
    block_data['y_int'] = _xy_to_int_msat_v04NA01(block_data['y'])

    df = block_data.merge(multisat_data, on=['x_int', 'y_int'], how='left')

    df = df[['block_id', 'exposure']]
    df = df.set_index('block_id')

    return df

def _xy_to_int_msat_v04NA01(x):
    return np.around(x * 100).astype(int)


if __name__ == '__main__':
    df = modis_exposure_bg_conus()
