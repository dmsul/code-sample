"""
Estimate excess mortality from being misclassified as "attainment" under clean
air act.

See Section 5.4.2 of "Using Satellite Data to Fill the Gaps in the US Air
Pollution Monitoring Network"
"""
import numpy as np
from econtools import state_fips_to_name

from util.env import out_path
from clean.mortality import mortality
from analysis.misclass import blocks_misclass_flag

from reg_nonattain import regs


DOSE_RATE = .14 / 10     # per 10 ug/m3 (Lepeule el al. 2012)
VSL = 9                  # Values in millions


def main(rule='pm25_12', data='msatna', save=False):

    # Full regression-based method
    ols, ols_w_flag, df_reg, __ = regs(rule=rule)

    df = prep_exposure_data(rule=rule, data=data)

    df['coeff'] = ols_w_flag.beta['untargeted_post']
    df.loc[df['is_over'], 'coeff'] = ols_w_flag.beta['targeted_post']

    df['extra_deaths'] = _dose_rate(df['coeff']) * df['block_deaths']
    extra_deaths = df['extra_deaths'].sum()
    targeted_deaths = df.loc[df['is_over'], 'extra_deaths'].sum()
    untargeted_deaths = df.loc[~df['is_over'], 'extra_deaths'].sum()
    df['state'] = df['fips'].apply(lambda x: state_fips_to_name(int(x[:2])))
    by_state = df.groupby('state')['extra_deaths'].sum().to_frame()
    by_state['cost'] = by_state['extra_deaths'] * VSL

    # Simple peak-shaving Method (lower to NAAQS only)
    df['to_naaqs'] = (df['exp'] - 12).clip_lower(0)
    df['to_naaqs_deaths'] = _dose_rate(df['to_naaqs']) * df['block_deaths']
    to_naaqs_deaths = df['to_naaqs_deaths'].sum()
    to_naaqs_decrease = df.loc[df['to_naaqs'] > 0, 'to_naaqs'].mean()

    # Scale whole county down by max
    df['county_max'] = df.groupby('fips')['exp'].transform('max')
    df['scale'] = 12 / df['county_max']
    df['scale_diff'] = df['exp'] * (1 - df['scale'])
    df['scale_deaths'] = _dose_rate(df['scale_diff']) * df['block_deaths']
    scale_deaths = df['scale_deaths'].sum()
    scale_decrease = df['scale_diff'].mean()

    # Output results
    out_str = (
        f"Extra deaths:\t{extra_deaths:.1f}\n" +
        "VSL value:\t{:.1f}\n".format(extra_deaths * VSL) +
        "\n" +
        f"In areas over NAAQS:\t{targeted_deaths:.1f}\n" +
        "VSL value:\t{:.1f}\n".format(targeted_deaths * VSL) +
        "\n" +
        f"Under NAAQS:\t{untargeted_deaths:.1f}\n" +
        "VSL value:\t{:.1f}\n".format(untargeted_deaths * VSL) +
        "\n" +
        by_state.to_string() +
        "\n---------------\n" +
        "Lowering to NAAQS only\n" +
        f"Extra deaths:\t{to_naaqs_deaths:.1f}\n" +
        "VSL value:\t{:.1f}\n".format(to_naaqs_deaths * VSL) +
        f"Decrease in exposure:\t{to_naaqs_decrease:.2f}\n" +
        "\n---------------\n" +
        "Scaling the peak down to 12\n" +
        f"Extra deaths:\t{scale_deaths:.1f}\n" +
        "VSL value\t{:.1f}\n".format(scale_deaths * VSL) +
        f"Decrease in exposure:\t{scale_decrease:.2f}\n"
    )
    print(out_str)
    if save:
        with open(out_path('calc_mortality.txt'), 'w') as f:
            f.write(out_str)

    return df


def _dose_rate(x):
    """ De-log that nonsense """
    return -1 * (1 - np.exp(DOSE_RATE * x))


def prep_exposure_data(rule='pm25_12', data='msatna'):
    exp_year = 2014 if rule == 'pm25_12' else 2007

    df = blocks_misclass_flag(exp_year, rule, data)

    # mortality
    mort = mortality()[['fips', 'year', 'deaths', 'rate']]
    mort = (mort[mort['year'] == exp_year]
            .set_index('fips')
            .drop('year', axis=1))
    df = df.join(mort, on='fips')

    df['block_deaths'] = df['pop'] * df['deaths'] / df['fips_pop']

    df = df[~df['nonattain']]

    has_misclass = df.groupby('fips')['is_over'].max()
    df = df.join(has_misclass.to_frame('has_misclass'), on='fips')

    df = df[df['has_misclass']]

    return df


if __name__ == '__main__':
    import argparse
    opts = argparse.ArgumentParser()
    opts.add_argument('--rule', type=str, default='pm25_12')
    opts.add_argument('--data', type=str, default='msatna',
                      choices=['multisatpm', 'msatna'])
    opts.add_argument('--save', action='store_true')
    args = opts.parse_args()

    df = main(args.rule, args.data, save=args.save)
