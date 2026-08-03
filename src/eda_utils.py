import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from IPython.display import display

# Ignore warning messages
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# Constants
# =============================================================================

PALETTE = {"yes": "#60E6A8", "no": "#E66176"}


# =============================================================================
# Functions
# =============================================================================

def column_summary(df, col):
    """
    Print basic statistics for a column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    col : str
        Name of the column to analyze.

    Notes
    -----
    - For categorical features, displays the total number of values,
      unique values, and missing values.
    - For numerical features, also prints the skewness.
    """

    print('=' * 20, f'Column: {col}', '=' * 20)
    print(f'Total number of values: {df[col].shape[0]}')
    print(f'Number of unique values: {df[col].nunique()}')
    print(f'Number of missing values: {df[col].isna().sum()}')

    if pd.api.types.is_numeric_dtype(df[col]):
        print(f'Skewness: {df[col].skew().round(2)}')

    print('=' * 50)


def null_analyze(df, col, verbose=True):
    """
    Analyze missing values in a column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    col : str
        Name of the column to analyze.
    verbose : bool, optional
        If True, prints the results. If False, only returns the DataFrame.

    Returns
    -------
    pandas.DataFrame
        Table containing the number and percentage of missing values.
    """

    null_count = df[col].isnull().sum()
    null_percentage = df[col].isnull().mean() * 100

    result = pd.DataFrame({
        'column': [col],
        'null_count': [null_count],
        'null_percentage': [null_percentage]
    })

    if verbose:
        print(result)
        print('=' * 50)

    return result


def eda_category(df, col, target, plots=True):
    """
    Perform exploratory data analysis (EDA) for a categorical feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    col : str
        Name of the categorical feature to analyze.
    target : str
        Name of the categorical target variable (e.g. "yes"/"no").
        Used for plotting and calculating percentage distributions.
    plots : bool, optional
        If True, generates visualizations.

    Notes
    -----
    - The target variable should be categorical (e.g. "yes"/"no")
      to ensure meaningful legends and plots.
    - The function also displays value counts and their percentages
      as part of the analysis.
    """

    # general information
    column_summary(df, col)

    # value counts and percentages
    value_counts_df = pd.DataFrame({
        'value_counts': df[col].value_counts(),
        'value_percentage': df[col].value_counts(normalize=True).round(4) * 100
    })
    display(value_counts_df.T)
    print('=' * 50)

    # missing value statistics (if any)
    if df[col].isna().sum():
        null_analyze(df, col, verbose=True)

    # visualizations
    if plots:
        palette = PALETTE
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)

        # count plot
        sns.countplot(
            data=df,
            x=col,
            hue=target,
            palette=palette,
            ax=axes[0]
        )
        axes[0].set_title(f'Count plot: {col}')
        axes[0].grid(axis='y', alpha=0.7)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(title=target)

        # percentage bar chart
        percent_df = df.groupby(col)[target].value_counts(normalize=True).unstack() * 100
        percent_df.plot(
            kind='bar',
            color=[PALETTE['no'], PALETTE['yes']],
            ax=axes[1]
        )

        axes[1].set_title('Percentage of yes/no for each category')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.7)

        plt.tight_layout()
        plt.show()

        # conversion plot
        yes_df = pd.crosstab(df[col], df[target], normalize='index') * 100

        plt.figure(figsize=(6, 4))
        yes_df['yes'].sort_values(ascending=False).plot(kind='bar', color="#60E6A8")

        plt.title(f'Conversion by {col}', fontsize=13)
        plt.xlabel(col)
        plt.ylabel('P(Yes), %')
        plt.grid(axis='y', alpha=0.7)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


def eda_numeric(df, col, target_col, plots=True):
    """
    Perform exploratory data analysis (EDA) for a numerical feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    col : str
        Name of the numerical feature to analyze.
    target_col : str, optional
        Name of the numerical target variable (e.g. 0/1).
        Used to calculate class medians and generate visualizations.
    plots : bool, optional
        If True, generates visualizations.

    Notes
    -----
    - The target variable should be numerical (0/1)
      to correctly calculate medians and summary statistics.
    - The function also estimates the number of outliers using the IQR rule
      and prints descriptive statistics for numerical features.
    """

    # general information
    column_summary(df, col)

    # outliers
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f'Outliers: {len(outliers)}')

    # class medians (no/yes)
    median_0 = df[df[target_col] == 0][col].median()
    median_1 = df[df[target_col] == 1][col].median()
    print(f'Median "no": {median_0}')
    print(f'Median "yes": {median_1}')
    print('=' * 50)

    print(df[col].describe().round(2))
    print('=' * 50)

    # missing value statistics (if any)
    if df[col].isna().sum():
        null_analyze(df, col, verbose=True)

    # visualizations
    if plots:
        palette = PALETTE
        labels = {0: 'no', 1: 'yes'}

        target_series = df[target_col].map(labels)

        plt.figure(figsize=(8, 8))

        # histogram
        plt.subplot(2, 1, 1)
        sns.histplot(data=df, x=col, hue=target_series, kde=True, palette=palette)
        plt.title(f'Histogram of {df[col].name}')
        plt.grid()

        # box plot
        plt.subplot(2, 2, 3)
        sns.boxplot(data=df, x=target_series, y=col, hue=target_series,
                    palette=palette)
        plt.title(f'Box plot of {df[col].name}')
        plt.grid()

        # violin plot
        plt.subplot(2, 2, 4)
        sns.violinplot(data=df, x=col, y=target_series, hue=target_series, palette=palette)
        plt.title(f'Violin plot of {df[col].name}')

        plt.tight_layout()
        plt.grid()
        plt.show()
