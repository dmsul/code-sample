from econtools import force_iterable


def weighted_quantile(df, var_name, wt_name, q=.5):
    """
    Returns weighted quantile of `var_name` given weights `wt_name`.

    `q` is the desired quantile, can be a single number or a list.
    """
    tmp_df = df[[var_name, wt_name]].sort_values(var_name)
    cumsum = tmp_df[wt_name].cumsum()

    q_iter = force_iterable(q)
    for x in q_iter:
        try:
            assert 0 < x < 1
        except AssertionError:
            raise ValueError("Quantiles must be between 0 and 1.")

    wt_sum = tmp_df[wt_name].sum()

    cutoffs = [wt_sum * x for x in q_iter]
    quantiles = [
        tmp_df.loc[(cumsum >= cutoff), var_name].iloc[0]
        for cutoff in cutoffs]

    if len(quantiles) == 1:
        return quantiles[0]
    else:
        return quantiles
