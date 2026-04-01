# Load Required Packages
import pandas as pd
import numpy as np


def iv_woe(data, target, bins=10, show_woe=False):
    """
    Calculate Information Value (IV) and Weight of Evidence (WoE) for variables.
    This function computes IV and WoE statistics for all independent variables in a dataset.
    Continuous variables with more than 10 unique values are binned, while categorical
    variables and those with fewer unique values are processed as-is.
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing both target and independent variables.
    target : str
        Name of the target variable column (binary classification).
    bins : int, optional
        Number of bins for continuous variables (default is 10).
        Applied only to numeric columns with more than 10 unique values.
    show_woe : bool, optional
        If True, prints the WoE table for each variable (default is False).
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - new_df : DataFrame with columns ['Variable', 'IV']
          Contains the Information Value for each variable.
        - woe_df : DataFrame with columns ['Variable', 'Cutoff', 'N', 'Events',
          '% of Events', 'Non-Events', '% of Non-Events', 'WoE', 'IV']
          Contains detailed WoE and IV statistics for each bin/category of each variable.
    Notes
    -----
    - A minimum count of 0.5 is applied to events and non-events to avoid zero divisions.
    - Variables are automatically excluded from processing.
    - Continuous variables are binned using quantile-based binning with duplicates dropped.
    """

    # Empty Dataframe
    new_df, woe_df = pd.DataFrame(), pd.DataFrame()

    # Extract Column Names
    cols = data.columns

    # Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
        d['IV'] = d['WoE'] * (d['% of Non-Events']-d['% of Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]},
                            columns=["Variable", "IV"])
        new_df = pd.concat([new_df, temp], axis=0)
        woe_df = pd.concat([woe_df, d], axis=0)
        # Show WOE Table
        if show_woe is True:
            print(d)

    return new_df, woe_df
