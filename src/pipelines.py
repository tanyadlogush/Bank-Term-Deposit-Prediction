import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    RobustScaler,
    OrdinalEncoder
)
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve
)



def create_preprocessor(numeric_cols, categorical_cols, scale_numeric=True, model_type=None):
    """
    Створює Pipeline з препроцесінгом.

    Параметри:
    - numeric_cols : список числових ознак
    - categorical_cols : список категоріальних ознак
    - scale_numeric : bool — чи масштабувати числові ознаки
    - model_type : str — тип моделію  Якщо 'lightgbm', категорії лишаються як є

    Повертає:
    - pipeline : sklearn Pipeline
    """

    # імпутація
    num_impute_step = ('imputer', SimpleImputer(strategy='median'))
    cat_impute_step = ('imputer', SimpleImputer(strategy='constant', fill_value='Missing'))

    # масштабування
    scale_step = ('scaler', RobustScaler())

    # кодування (OrdinalEncoder для LightGBM, encode_step - для інших алгоритмів)
    ordinal_step = ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    encode_step = ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))


    # трансформер для числових ознак
    if scale_numeric:
        numeric_transformer = Pipeline(steps=[num_impute_step, scale_step])
    else:
        numeric_transformer = Pipeline(steps=[num_impute_step])


    # трансформер для категоріальних ознак
    if model_type == 'lightgbm':
        categorical_transformer = Pipeline(steps=[cat_impute_step, ordinal_step])
    else:
        categorical_transformer = Pipeline(steps=[cat_impute_step, encode_step])

    # комбінація трансформерів
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor


def create_pipeline(model, numeric_cols, categorical_cols, model_type=None, scale_numeric=True):
    """
    Створює повний Pipeline: препроцесинг + модель.

    Параметри:
    - model : sklearn-сумісна модель
    - numeric_cols : список числових ознак
    - categorical_cols : список категоріальних ознак
    - model_type : str або None — тип моделі:
        'logistic', 'knn', 'tree', 'xgbm', 'lightgbm'
        Використовується для вибору способу обробки категоріальних ознак
    - scale_numeric : bool — чи застосовувати масштабування до числових ознак

    Повертає:
    - pipeline : sklearn Pipeline (preprocessor + model)
    """

    valid_model_types = {None, 'logistic', 'knn', 'tree', 'xgbm', 'lightgbm'}
    if model_type not in valid_model_types:
        raise ValueError(f"Unknown model_type: {model_type}")

    preprocessor = create_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=scale_numeric,
        model_type=model_type
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline



def evaluate_model(model_name, model, X_train, y_train, X_val, y_val, params=None, comments=None, results_table=None):
    """
    Оцінює модель та додає результати в таблицю.

    Параметри:
    - model_name : назва моделі (str)
    - model : навчена модель
    - X_train, y_train : тренувальні дані
    - X_val, y_val : валідаційні дані
    - params : dict — гіперпараметри моделі
    - comments : str — коментар до моделі
    - results_table : pd.DataFrame — таблиця з результатами (може бути None)

    Повертає:
    - results_table : оновлена таблиця з результатами
    """

    # прогнози
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # перевірка на наявність predict_proba у моделі
    if hasattr(model, 'predict_proba'):
        y_val_proba = model.predict_proba(X_val)[:, 1]
    else:
        raise ValueError(f'{model_name} does not support predict_proba')

    # ймовірності для AUROC
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]

    # метрики
    train_auc = roc_auc_score(y_train, y_train_proba)
    val_auc = roc_auc_score(y_val, y_val_proba)

    train_f1 = f1_score(y_train, y_train_pred)
    val_f1 = f1_score(y_val, y_val_pred)

    # додаткові звіти
    print(f'\n=== {model_name} ===')
    print('Classification report (validation):')
    print(classification_report(y_val, y_val_pred))
    print('Confusion matrix (validation):')
    print(confusion_matrix(y_val, y_val_pred))
    print(f'Train AUROC: {train_auc:.4f}')
    print(f'Validation AUROC: {val_auc:.4f}')

    # ROC-крива
    fpr, tpr, _ = roc_curve(y_val, y_val_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC={val_auc:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # таблиця результатів
    row = {
        'Model': model_name,
        'Params': params,
        'Train_AUROC': round(train_auc, 4),
        'Val_AUROC': round(val_auc, 4),
        'Train_F1': round(train_f1, 4),
        'Val_F1': round(val_f1, 4),
        'Comments': comments
    }

    if results_table is None:
        results_table = pd.DataFrame([row])
    else:
        results_table = pd.concat([results_table, pd.DataFrame([row])], ignore_index=True)

    return results_table

def prepare_data(train_df, val_df, input_cols, target_col, drop_cols=None):
    """
    Формує X_train, y_train, X_val, y_val та списки числових і категоріальних ознак.

    Параметри:
    - train_df, val_df : DataFrame — тренувальний та валідаційний набори
    - input_cols : список ознак для використання
    - target_col : назва цільової змінної
    - drop_cols : список колонок, які треба видалити (наприклад, ['duration'])

    Повертає:
    - X_train, y_train, X_val, y_val, num_features, cat_features
    """

    # видалення непотрібних колонок (за необхідності)
    if drop_cols:
        cols = [col for col in input_cols if col not in drop_cols]

    # формування тренувального та валідаційного наборів даних
    X_train = train_df[cols].copy()
    y_train = train_df[target_col].copy()

    X_val = val_df[cols].copy()
    y_val = val_df[target_col].copy()

    # списки числових та категоріальних колонок
    num_features = X_train.select_dtypes(include='number').columns.tolist()
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    return X_train, y_train, X_val, y_val, num_features, cat_features


import os

def save_results(results_table, filename="results.csv"):
    """
    Зберігає результати моделей у CSV-файл.

    Параметри:
    - results_table : pd.DataFrame — таблиця з результатами моделей
    - filename : str — ім'я файлу для збереження

    Логіка:
    - якщо файл не існує — створюється новий файл із усією таблицею
    - якщо файл існує — додається лише останній рядок (результат останнього експерименту)

    Примітка:
    - якщо results_table порожній або None, файл не створюється
    """

    if results_table is None or results_table.empty:
        print('Warning: Таблиця результатів пуста.')
        return

    if not os.path.exists(filename):
        results_table.to_csv(filename, index=False)
    else:
        # додаємо нові рядки без перезапису всього файлу
        results_table.tail(1).to_csv(filename, mode='a', header=False, index=False)
