import os

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler
)


def create_preprocessor(numeric_cols, categorical_cols, scale_numeric=True, model_type=None):
    """
    Create a preprocessing pipeline.

    Parameters:
    - numeric_cols : list of numerical features
    - categorical_cols : list of categorical features
    - scale_numeric : bool — whether to scale numerical features
    - model_type : str — model type. If 'lightgbm', categorical features are kept as ordinal values

    Returns:
    - pipeline : sklearn Pipeline
    """

    # imputation
    num_impute_step = ('imputer', SimpleImputer(strategy='median'))
    cat_impute_step = ('imputer', SimpleImputer(strategy='constant', fill_value='Missing'))

    # scaling
    scale_step = ('scaler', RobustScaler())

    # encoding (OrdinalEncoder for LightGBM, encode_step for other algorithms)
    ordinal_step = ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    encode_step = ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

    # transformer for numerical features
    if scale_numeric:
        numeric_transformer = Pipeline(steps=[num_impute_step, scale_step])
    else:
        numeric_transformer = Pipeline(steps=[num_impute_step])

    # transformer for categorical features
    if model_type == 'lightgbm':
        categorical_transformer = Pipeline(steps=[cat_impute_step, ordinal_step])
    else:
        categorical_transformer = Pipeline(steps=[cat_impute_step, encode_step])

    # combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor


def create_pipeline(model, numeric_cols, categorical_cols, model_type=None, scale_numeric=True):
    """
    Create a complete pipeline consisting of preprocessing and a model.

    Parameters:
    - model : sklearn-compatible model
    - numeric_cols : list of numerical features
    - categorical_cols : list of categorical features
    - model_type : str or None — model type:
        'logistic', 'knn', 'tree', 'xgbm', 'lightgbm'
        Used to determine how categorical features should be processed
    - scale_numeric : bool — whether to scale numerical features

    Returns:
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


def prepare_data(train_df, val_df, input_cols, target_col, drop_cols=None):
    """
    Create training and validation feature/target sets and identify numerical and categorical features.

    Parameters:
    - train_df, val_df : DataFrame — training and validation datasets
    - input_cols : list of input features
    - target_col : target column name
    - drop_cols : list of columns to remove (e.g. ['duration'])

    Returns:
    - X_train, y_train, X_val, y_val, num_features, cat_features
    """

    # remove unnecessary columns (if specified)
    if drop_cols:
        cols = [col for col in input_cols if col not in drop_cols]

    # create training and validation datasets
    X_train = train_df[cols].copy()
    y_train = train_df[target_col].copy()

    X_val = val_df[cols].copy()
    y_val = val_df[target_col].copy()

    # lists of numerical and categorical features
    num_features = X_train.select_dtypes(include='number').columns.tolist()
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    return X_train, y_train, X_val, y_val, num_features, cat_features


def evaluate_model(model_name, model, X_train, y_train, X_val, y_val, params=None, comments=None, results_table=None):
    """
    Evaluate a trained model and add the results to the results table.

    Parameters:
    - model_name : model name (str)
    - model : trained model
    - X_train, y_train : training data
    - X_val, y_val : validation data
    - params : dict — model hyperparameters
    - comments : str — additional comments about the model
    - results_table : pd.DataFrame — results table (can be None)

    Returns:
    - results_table : updated results table
    """

    # predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # check whether the model supports predict_proba
    if hasattr(model, 'predict_proba'):
        y_val_proba = model.predict_proba(X_val)[:, 1]
    else:
        raise ValueError(f'{model_name} does not support predict_proba')

    # prediction probabilities for AUROC
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]

    # metrics
    train_auc = roc_auc_score(y_train, y_train_proba)
    val_auc = roc_auc_score(y_val, y_val_proba)

    train_f1 = f1_score(y_train, y_train_pred)
    val_f1 = f1_score(y_val, y_val_pred)

    # additional reports
    print(f'\n=== {model_name} ===')
    print('Classification report (validation):')
    print(classification_report(y_val, y_val_pred))
    print('Confusion matrix (validation):')
    print(confusion_matrix(y_val, y_val_pred))
    print(f'Train AUROC: {train_auc:.4f}')
    print(f'Validation AUROC: {val_auc:.4f}')

    # ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_val_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC={val_auc:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # result table
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


def save_results(results_table, filename="results.csv"):
    """
    Save model evaluation results to a CSV file.

    Parameters:
    - results_table : pd.DataFrame — table containing model evaluation results
    - filename : str — output CSV file name

    Logic:
    - if the file does not exist, create a new file containing the entire table
    - if the file already exists, append only the last row (the most recent experiment)

    Note:
    - if results_table is None or empty, no file will be created
    """

    if results_table is None or results_table.empty:
        print('Warning: Results table is empty.')
        return

    if not os.path.exists(filename):
        results_table.to_csv(filename, index=False)
    else:
        # append only the latest result without overwriting the entire file
        results_table.tail(1).to_csv(filename, mode='a', header=False, index=False)
