import pandas as pd

from util.env import src_path


def mortality():
    """ https://wonder.cdc.gov/cmf-icd10.html """
    df = pd.read_table(src_path('Compressed Mortality, 1999-2016.txt'))
    df = df.rename(columns=lambda x: x.lower().replace(' ', '_'))
    df = df.rename(columns={'county_code': 'fips'})

    # Get rid of footnote garbage
    df = df[df['county'].notnull()].copy()

    assert df[df['notes'].notnull()].empty
    del df['notes']

    # Convert rate from string to float
    df['rate'] = df['crude_rate'].apply(
        lambda x: float(x.replace(' (Unreliable)', '')))

    df = df.drop(['year_code', 'county', 'crude_rate'], axis=1)

    for col in ('fips', 'year', 'deaths', 'population'):
        df[col] = df[col].astype(int)

    df['fips'] = df['fips'].astype(str).str.zfill(5)

    return df


if __name__ == '__main__':
    df = mortality()
