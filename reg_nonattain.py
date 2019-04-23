"""
Estimate the effect of regulator behavior on pollution concentrations at
monitors using Difference-in-difference regressions.

Table 4 in "Using Satellite Data to Fill the Gaps in the US Air Pollution
Monitoring Network"
"""
import econtools.metrics as mt
from econtools import outreg, table_statrow, save_cli, write_notes

from util.env import out_path
from analysis.monitor_sample import (constant_monitor_panel,
                                     prep_monitor_analysis,)


def main(rule='pm25_12', save=False):
    ols, ols_w_flag, df, _I = regs(rule=rule)

    table_str = make_table(ols, ols_w_flag, _I)

    # Table notes
    df['cat'] = (
        df['targeted'] * 1 +
        df['untargeted'] * 2 +
        (~df['nonattain']) * 3
    )
    summ = df.groupby(['cat', 'year'])['nonattain'].size().unstack('year')
    cat_N = summ[2015].tolist()

    notes = (
        f"N=\\num{{{ols_w_flag.N}}}.\n"
        "Number of monitors in Groups I, II, and III are"
        f" {cat_N[0]}, {cat_N[1]}, and {cat_N[2]}, respectively.\n"
    )

    print(table_str)
    print(notes)

    if save:
        filepath = out_path('mortality_regression_results.tex')
        with open(filepath, 'w') as f:
            f.write(table_str)
        write_notes(notes, filepath)

    return ols, ols_w_flag


def regs(rule='pm25_12'):
    df = constant_monitor_panel(rule=rule)
    df = prep_monitor_analysis(df, rule=rule)

    # Get list of control variables
    _I = df.filter(like='_I').columns.tolist()

    # Naive OLS
    ols = mt.reg(df, 'arithmetic_mean',
                     ['nonattain_post'] + _I,
                     a_name='monitor_id',
                     cluster='monitor_id')

    # Diff-in-diff
    z_vars = ['targeted_post', 'untargeted_post']
    ols_w_flag = mt.reg(df, 'arithmetic_mean',
                        z_vars + _I,
                        a_name='monitor_id',
                        cluster='monitor_id')

    return ols, ols_w_flag, df, _I


def make_table(ols, ols_w_flag, _I):
    var_names = (
        'nonattain_post',
        'targeted_post',
        'untargeted_post',
    ) + tuple(_I)
    var_labels = [
        r'Nonattainment$\times$post',
        r'Nonattainment$\times$Over NAAQS$\times$post',
        r'Nonattainment$\times$Under NAAQS$\times$post',
    ] + [x[-4:] for x in _I]

    table_str = outreg((ols, ols_w_flag), var_names, var_labels)
    table_str += '\\\\\n'
    table_str += table_statrow("R$^2$", [reg.r2 for reg in (ols, ols_w_flag)],
                               digits=3)

    return table_str


if __name__ == "__main__":
    ols, ols_w_flag, df, _I = main(save=save_cli())
