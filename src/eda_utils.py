import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from IPython.display import display

# Налаштування для ігнорування попереджень
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Константи для візуалізації
PALETTE = {"yes": "#60E6A8", "no": "#E66176"}



def column_summary(df, col):
    """
    Друкує базову статистику для колонки.

    Parameters
    ----------
    df : pandas.DataFrame
        Вхідний датафрейм.
    col : str
        Назва колонки для аналізу.

    Notes
    -----
    - Для категоріальних ознак показує кількість значень, унікальних та пропусків.
    - Для числових ознак додатково виводить асиметрію (skewness).
    """
    print('=' * 20, f'Column: {col}', '=' * 20)
    print(f'Загальна кількість значень: {df[col].shape[0]}')
    print(f'Кількість унікальних значень: {df[col].nunique()}')
    print(f'Кількість пропущених значень: {df[col].isna().sum()}')

    if pd.api.types.is_numeric_dtype(df[col]):
        print(f'Асиметрія даних: {df[col].skew().round(2)}')

    print('=' * 50)


def null_analyze(df, col, verbose=True):
    """
    Аналіз пропущених значень у колонці.

    Parameters
    ----------
    df : pandas.DataFrame
        Вхідний датафрейм.
    col : str
        Назва колонки для аналізу.
    verbose : bool, optional
        Якщо True, друкує результати. Якщо False, лише повертає DataFrame.

    Returns
    -------
    pandas.DataFrame
        Таблиця з кількістю та відсотком пропущених значень.
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
      Проведення EDA для категоріальної ознаки.

      Parameters
      ----------
      df : pandas.DataFrame
          Вхідний датафрейм.
      col : str
          Назва категоріальної ознаки для аналізу.
      target : str
          Назва цільової змінної у категоріальному вигляді (наприклад, "yes"/"no").
          Використовується для побудови графіків та розрахунку процентних співвідношень.
      plots : bool, optional
          Якщо True, будуються графіки.

      Notes
      -----
      - Цільова змінна тут має бути саме категоріальною (наприклад, "yes"/"no"),
        щоб легенди та графіки були зрозумілими.
      - Функція додатково виводить розподіл значень (value_counts) та їх частки»,
        щоб було зрозуміло, що це частина логіки.
      """
    # загальні дані
    column_summary(df, col)

    # таблиця значень та їх частки
    value_counts_df = pd.DataFrame({
        'value_counts': df[col].value_counts(),
        'value_percentage': df[col].value_counts(normalize=True).round(4) * 100
    })
    display(value_counts_df.T)
    print('=' * 50)

    # статистика пропущених значень (за умови їх наявності)
    if df[col].isna().sum():
        null_analyze(df, col, verbose=True)

    # візуалізація
    if plots:
        palette = PALETTE
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)

        # гістограма
        sns.countplot(
            data=df,
            x=col,
            hue=target,
            palette=palette,
            ax=axes[0]
        )
        axes[0].set_title(f'Стовпчата діаграма: {col}')
        axes[0].grid(axis='y', alpha=0.7)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(title=target)

        # діаграма розмаху
        percent_df = df.groupby(col)[target].value_counts(normalize=True).unstack() * 100
        percent_df.plot(
            kind='bar',
            color=[PALETTE['no'], PALETTE['yes']],
            ax=axes[1]
        )

        axes[1].set_title('Процентне співвідношення yes/no по кожній категорії')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.7)

        plt.tight_layout()
        plt.show()

        # графік конверсії
        yes_df = pd.crosstab(df[col], df[target], normalize='index') * 100

        plt.figure(figsize=(6, 4))
        yes_df['yes'].sort_values(ascending=False).plot(kind='bar', color="#60E6A8")

        plt.title(f'Конверсія {col}', fontsize=13)
        plt.xlabel(col)
        plt.ylabel('P(Yes), %')
        plt.grid(axis='y', alpha=0.7)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


def eda_numeric(df, col, target_col, plots=True):
    """
      Проведення EDA для числової ознаки.

      Parameters
      ----------
      df : pandas.DataFrame
          Вхідний датафрейм.
      col : str
          Назва числової ознаки для аналізу.
      target_col : str, optional
          Назва цільової змінної у числовому вигляді (наприклад, 0/1).
          Використовується для розрахунку медіан та побудови графіків розподілу.
      plots : bool, optional
          Якщо True, будуються графіки.

      Notes
      -----
      - Цільова змінна тут має бути саме числовою (0/1),
        щоб коректно рахувати медіани та статистику.
      - Функція додатково оцінює кількість викидів за правилом IQR,
        а також друкує описову статистику - цей блок використовується
        лише для числових ознак.
      """

    # загальні дані
    column_summary(df, col)

    # викиди
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f'Викиди: {len(outliers)}')

    # медіани no/yes
    median_0 = df[df[target_col] == 0][col].median()
    median_1 = df[df[target_col] == 1][col].median()
    print(f'Медіана "no": {median_0}')
    print(f'Медіана "yes": {median_1}')
    print('=' * 50)

    print(df[col].describe().round(2))
    print('=' * 50)

    # статистика пропущених значень  (за умови їх наявності)
    if df[col].isna().sum():
        null_analyze(df, col, verbose=True)

    # Візуалізація
    if plots:
        palette = PALETTE
        labels = {0: 'no', 1: 'yes'}

        target_series = df[target_col].map(labels)

        plt.figure(figsize=(8, 8))

        # гістограма
        plt.subplot(2, 1, 1)
        sns.histplot(data=df, x=col, hue=target_series, kde=True, palette=palette)
        plt.title(f'Гістограма {df[col].name}')
        plt.grid()

        # діаграма
        plt.subplot(2, 2, 3)
        sns.boxplot(data=df, x=target_series, y=col, hue=target_series,
                    palette=palette)
        plt.title(f'Діаграма розмаху {df[col].name}')
        plt.grid()

        # violin-діаграма
        plt.subplot(2, 2, 4)
        sns.violinplot(data=df, x=col, y=target_series, hue=target_series, palette=palette)
        plt.title(f'Violin-діаграма розмаху {df[col].name}')

        plt.tight_layout()
        plt.grid()
        plt.show()