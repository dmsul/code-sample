"""
Estimate the event study version of regressions in `reg_nonattain.py`.

Figure 7 in "Using Satellite Data to Fill the Gaps in the US Air Pollution
Monitoring Network"
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import econtools.metrics as mt
from econtools import legend_below, save_cli


from util.env import out_path
from analysis.monitor_sample import (constant_monitor_panel,
                                     prep_monitor_analysis,)


if __name__ == "__main__":
    # Prep data
    rule = 'pm25_12'
    df = constant_monitor_panel(rule=rule)
    df = prep_monitor_analysis(df, rule=rule)

            # Drop outlier
    df = df[df['monitor_id'] != '06031_4_881011']

    # Create treatment-year interactions
    z_vars = ('targeted', 'untargeted')
    interact_years = [x for x in df['year'].unique() if x != 2015]
    for z_var in z_vars:
        for y in interact_years:
            df[f'z_{z_var}_{y}'] = (df['year'] == y) & (df[z_var])

    X = (df.filter(like='z_').columns.tolist() +
         df.filter(like='_I').columns.tolist())

    # Event study regression
    res = mt.reg(df, 'arithmetic_mean', X,
                 a_name='monitor_id',
                 cluster='monitor_id')
    print(res)

    # Plot event study coefficients
    betas = pd.DataFrame(index=interact_years)
    ci_hi = pd.DataFrame(index=interact_years)
    ci_lo = pd.DataFrame(index=interact_years)
    for z_var in z_vars:
        betas[z_var] = res.beta.filter(like=f'z_{z_var}').values
        ci_hi[z_var] = res.ci_hi.filter(like=f'z_{z_var}').values
        ci_lo[z_var] = res.ci_lo.filter(like=f'z_{z_var}').values

    omitted_year = 2015
    betas.loc[omitted_year, :] = 0
    ci_hi.loc[omitted_year, :] = np.nan
    ci_lo.loc[omitted_year, :] = np.nan

    betas = betas.sort_index()
    ci_hi = ci_hi.sort_index()
    ci_lo = ci_lo.sort_index()

    fig, ax = plt.subplots()
    styles = dict(zip(z_vars, (
        {
            'label': r'Nonattainment Over NAAQS (Group I)',
            'color': 'maroon',
            'linestyle': '-',
            'marker': 'o'
        }, {
            'label': r'Nonattainment Under NAAQS (Group II)',
            'color': 'orange',
            'linestyle': '--',
            'marker': '^'
        })))
    for z_var in z_vars:
        ax.errorbar(betas.index, betas[z_var],
                    yerr=(np.abs(betas[z_var] - ci_hi[z_var])),
                    capsize=3,
                    **styles[z_var],
                    )
    ax.axhline(0, color='k', linewidth=.8)              # 0 line
    ax.axvline(omitted_year,                            # Treatment line
               color='g', linestyle=':', zorder=2)
    ax.yaxis.grid(True, alpha=.5)
    ax.set_ylim(np.floor(ci_lo.min().min()), np.ceil(ci_hi.max().max()))

    ax.set_ylabel(
        "Change in PM$_{2.5}$ "
        "Relative to Attainment (Group III)"
    )
    ax.set_xlabel("Year")
    ax.set_xticks(betas.index)

    legend_below(ax, ncol=1)

    if save_cli():
        filepath = out_path(f'reg_es.pdf')
        fig.savefig(filepath, bbox_inches='tight', transparent=True)
        fig.savefig(filepath.replace('.pdf', '.png'),
                    bbox_inches='tight',
                    dpi=400,
                    transparent=True
                    )
        plt.close()
    else:
        plt.show()
