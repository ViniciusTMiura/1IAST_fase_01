# Load Required Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
        if (data[ivars].dtype.kind in "bifc") and (len(np.unique(data[ivars])) > 10):
            binned_x = pd.qcut(data[ivars], bins, duplicates="drop")
            d0 = pd.DataFrame({"x": binned_x, "y": data[target]})
        else:
            d0 = pd.DataFrame({"x": data[ivars], "y": data[target]})
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ["Cutoff", "N", "Events"]
        d["% of Events"] = np.maximum(d["Events"], 0.5) / d["Events"].sum()
        d["Non-Events"] = d["N"] - d["Events"]
        d["% of Non-Events"] = np.maximum(d["Non-Events"], 0.5) / d["Non-Events"].sum()
        d["WoE"] = np.log(d["% of Non-Events"] / d["% of Events"])
        d["IV"] = d["WoE"] * (d["% of Non-Events"] - d["% of Events"])
        d.insert(loc=0, column="Variable", value=ivars)
        print("Information value of " + ivars + " is " + str(round(d["IV"].sum(), 6)))
        temp = pd.DataFrame(
            {"Variable": [ivars], "IV": [d["IV"].sum()]}, columns=["Variable", "IV"]
        )
        new_df = pd.concat([new_df, temp], axis=0)
        woe_df = pd.concat([woe_df, d], axis=0)
        # Show WOE Table
        if show_woe is True:
            print(d)

    return new_df, woe_df


def print_correlacao(df):
    """
    Generates and displays a correlation heatmap for numerical columns in a DataFrame.
    This function calculates the Pearson correlation matrix for all numerical columns
    in the input DataFrame and visualizes it as a color-coded heatmap. The visualization
    uses a diverging color palette (blue to red) to emphasize positive and negative
    correlations, with the lower triangle of the correlation matrix displayed.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing numerical and/or non-numerical columns.
        Only numerical columns will be used for correlation calculation.
    Returns
    -------
    dict
        A dictionary containing summary information about the correlation analysis:
        - 'arquivo' (str): Suggested filename for the generated visualization
        - 'dimensoes' (tuple): Shape of the correlation matrix (number of features)
        - 'amostra_corr' (list): Sample of first 3 correlation entries as key-value pairs
    Notes
    -----
    - The correlation method used is Pearson correlation coefficient
    - Only the lower triangle of the correlation matrix is displayed
    - Correlation values are centered at 0 with range [-1, 1]
    - The heatmap includes annotations showing correlation values rounded to 2 decimal places
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('data.csv')
    >>> result = print_correlacao(df)
    >>> print(result['dimensoes'])
    """

    # 1) Selecionar apenas colunas numéricas
    num_df = df.select_dtypes(include=[np.number])

    # 2) Calcular a matriz de correlação (Pearson por padrão)
    corr = num_df.corr(method="pearson")

    # 3) Plotar como heatmap com cores por intensidade
    plt.figure(figsize=(12, 12))
    # Colormap divergente enfatiza sinais positivos/negativos
    cmap = sns.diverging_palette(240, 10, as_cmap=True)  # azul->vermelho

    # Máscara para mostrar apenas triângulo inferior (opcional)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    ax = sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,  # intensidade fixa em [-1, 1]
        center=0,
        annot=True,
        fmt=".2f",  # anotações opcionais
        linewidths=0.5,
        cbar_kws={"label": "Correlação de Pearson"},
    )
    ax.set_title("Mapa de calor das correlações (intensidade de cor)")
    plt.tight_layout()

    # Retornar algumas informações úteis
    corr_summary = corr.round(2).to_dict()
    artifacts = {
        "arquivo": "correlation_heatmap_exemplo.png",
        "dimensoes": corr.shape,
        "amostra_corr": list(corr_summary.items())[:3],
    }

    plt.show()

    return artifacts


def analise_quantil(df, col, n=10):
    """
    Analyze the relationship between a continuous variable and NPS score using quantiles.
    This function divides a specified column into quantiles, calculates
    the mean NPS score for each quantiles, and visualizes the results in a line plot.
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing the data to analyze.
    col : str
        The name of the column to be divided into quantiles.
    n : int, optional
        The number of quantiles to create (default is 10 for quantiles).
        If there are duplicate values at quantile boundaries, the number of
        groups may be less than n.
    Returns
    -------
    None
        Displays a plot and prints quantile limits to console.
    Notes
    -----
    - The function modifies a copy of the input dataframe, leaving the original unchanged.
    - Duplicate values at quantile boundaries are handled by the 'drop' strategy.
    - The plot uses a viridis color palette with a whitegrid style.
    - The function assumes the dataframe contains an 'nps_score' column.
    Examples
    --------
    >>> analise_quantil(df, 'age', n=10)
    # Displays quantile limits and a line plot of mean NPS score by age quantiles
    """
    # Criar quantil
    df["quantil"] = pd.qcut(df[col], n, labels=False, duplicates="drop") + 1

    limites = df.groupby("quantil")[col].agg(min_val="min", max_val="max")

    print(limites)

    # Agregar dados
    df_plot = df.groupby("quantil", as_index=False).agg(nps_medio=("nps_score", "mean"))

    # Plot
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df_plot, x="quantil", y="nps_medio", marker="o", linewidth=2)

    # Ajustes visuais
    plt.title(f"NPS Médio por Quantil da variável {col}", fontsize=16, weight="bold")
    plt.xlabel("Quantil", fontsize=12)
    plt.ylabel("NPS Médio", fontsize=12)
    plt.xticks(range(1, df_plot["quantil"].max() + 1))
    plt.ylim(0, 10)

    plt.tight_layout()
    plt.show()

    df.drop(columns=["quantil"], inplace=True)
    return None
