import errno
import logging
import os
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO)


class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # get month
        X["SAFRA"] = X["SAFRA"].str[:-2]
        return X


def data_dtype() -> Dict:
    return {
        "SAFRA": str,
        "V1": "float32",
        "V2": "float32",
        "V3": "float32",
        "V4": "float32",
        "V5": "float32",
        "V6": "float32",
        "V7": "float32",
        "V8": "float32",
        "V9": "float32",
        "V10": "float32",
        "V11": str,
        "V12": str,
        "CEP": str,
        "TARGET": int,
    }


def read_data(data_path: Path) -> pd.DataFrame:
    if not data_path.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(data_path))
    logging.info(f"reading file: {data_path}")
    return pd.read_csv(
        data_path,
        sep=",",
        header=0,
        encoding="latin-1",
        dtype=data_dtype(),
    )


def build_model() -> BaseEstimator:
    numeric_features = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V7", "V9", "V10"]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_features = ["V11", "V12"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    date_features = ["SAFRA"]
    date_transformer = DateTransformer()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("date", date_transformer, date_features),
        ]
    )
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("sampling", SMOTE()),
            ("classifier", MultinomialNB(force_alpha=True)),
        ]
    )
    return clf


if __name__ == "__main__":
    project_folder: Path = Path(__file__).parent.parent
    data_folder: Path = project_folder / "data"
    output_folder: Path = project_folder / "ml"
    output_folder.mkdir(parents=True, exist_ok=True)

    raw_df = read_data(data_folder / "dataset_test_ds_v2-Atualizado.csv")
    clf_pipeline: BaseEstimator = build_model()

    X, y = raw_df[raw_df.columns[:-1]], raw_df[raw_df.columns[-1]]

    with open(data_folder / "StratifiedKFold.pkl", "rb") as f:
        skf = pickle.load(f)

    best_recall = 0
    for train_index, val_index in skf.split(X, y):
        x_train_fold, x_val_fold = X.loc[train_index], X.loc[val_index]
        y_train_fold, y_val_fold = y.loc[train_index], y.loc[val_index]
        clf_pipeline.fit(x_train_fold, y_train_fold)
        y_pred = clf_pipeline.predict(x_val_fold)
        bal_acc = balanced_accuracy_score(y_val_fold, y_pred)
        recall = recall_score(y_val_fold, y_pred)
        if recall > best_recall:
            best_recall = recall
            dump(clf_pipeline, output_folder / "clf.joblib")
            logging.info("saved the model with results below:")
        logging.info("    balanced_accuracy_score: %.3f" % bal_acc)
        logging.info("    recall_score: %.3f\n" % recall)
