"""Shared data preparation and model-building utilities for the MLDP project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "hdb.csv"

SELECTED_TOWNS = ["ANG MO KIO", "BISHAN", "TOA PAYOH"]

CATEGORICAL_FEATURES = [
    "town",
    "flat_type",
    "street_name",
    "flat_model",
]

NUMERICAL_FEATURES = [
    "floor_area_sqm",
    "remaining_lease_years",
    "storey_mid",
    "sale_year",
    "sale_month",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "resale_price"

BEST_PARAMETERS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5,
}


class PandasDummyEncoder(BaseEstimator, TransformerMixin):
    """Apply the lecture's pd.get_dummies method consistently.

    Categories and output columns are learned from the training set during
    fit. Test and prediction data are then aligned to those same columns,
    preventing column mismatches and avoiding train-test leakage.
    """

    def __init__(self, categorical_features: list[str] | None = None):
        self.categorical_features = categorical_features

    def fit(self, features: pd.DataFrame, target=None):
        frame = features.copy()
        categorical_features = list(self.categorical_features or [])
        missing = [
            column
            for column in categorical_features
            if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                f"Categorical columns not found: {', '.join(missing)}"
            )

        self.categories_ = {
            column: sorted(frame[column].dropna().astype(str).unique())
            for column in categorical_features
        }
        encoded = self._encode(frame)
        self.feature_names_out_ = encoded.columns.to_list()
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["categories_", "feature_names_out_"])
        encoded = self._encode(features.copy())
        return encoded.reindex(
            columns=self.feature_names_out_,
            fill_value=0.0,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)

    def _encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        for column, categories in self.categories_.items():
            frame[column] = pd.Categorical(
                frame[column].astype(str),
                categories=categories,
            )
        return pd.get_dummies(
            frame,
            columns=list(self.categories_),
            drop_first=True,
            dtype=float,
        )


def convert_lease_to_years(lease: str) -> float:
    """Convert strings such as '61 years 04 months' to decimal years."""
    parts = str(lease).split()
    years = 0
    months = 0

    if "years" in parts:
        years = int(parts[parts.index("years") - 1])
    elif "year" in parts:
        years = int(parts[parts.index("year") - 1])

    if "months" in parts:
        months = int(parts[parts.index("months") - 1])
    elif "month" in parts:
        months = int(parts[parts.index("month") - 1])

    return years + months / 12


def prepare_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Apply the same cleaning, scope and feature engineering as the notebook."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataframe = pd.read_csv(data_path)
    dataframe = dataframe.drop_duplicates().copy()
    dataframe = dataframe[dataframe["town"].isin(SELECTED_TOWNS)].copy()

    storey_bounds = dataframe["storey_range"].str.extract(
        r"(?P<lower>\d+) TO (?P<upper>\d+)"
    )
    if storey_bounds.isna().any().any():
        raise ValueError("One or more storey ranges could not be converted.")

    dataframe["storey_mid"] = (
        storey_bounds.astype(float).mean(axis=1)
    )
    dataframe["remaining_lease_years"] = dataframe[
        "remaining_lease"
    ].apply(convert_lease_to_years)
    dataframe["sale_year"] = dataframe["month"].str[:4].astype(int)
    dataframe["sale_month"] = dataframe["month"].str[5:7].astype(int)

    required_columns = MODEL_FEATURES + [TARGET, "storey_range"]
    if dataframe[required_columns].isna().any().any():
        raise ValueError("Prepared data contains missing model values.")

    return dataframe


def build_tuned_pipeline() -> Pipeline:
    """Create the tuned Random Forest pipeline selected in the notebook."""
    preprocessor = PandasDummyEncoder(CATEGORICAL_FEATURES)

    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        **BEST_PARAMETERS,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def sort_storey_ranges(values: list[str]) -> list[str]:
    """Sort HDB storey labels by their lower storey number."""
    return sorted(values, key=lambda value: int(value.split()[0]))


def storey_midpoint(storey_range: str) -> float:
    """Convert an HDB storey label such as '10 TO 12' to its midpoint."""
    parts = storey_range.split()
    if len(parts) != 3 or parts[1] != "TO":
        raise ValueError("Please select a valid storey range.")
    return (float(parts[0]) + float(parts[2])) / 2
