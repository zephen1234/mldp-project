"""Train and export the final tuned model used by the Streamlit application."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from model_utils import (
    BEST_PARAMETERS,
    MODEL_FEATURES,
    PROJECT_DIR,
    TARGET,
    build_tuned_pipeline,
    prepare_data,
    sort_storey_ranges,
)


MODEL_PATH = PROJECT_DIR / "hdb_price_pipeline.joblib"
METADATA_PATH = PROJECT_DIR / "model_metadata.json"


def nested_options(dataframe, group_column: str, value_column: str) -> dict:
    """Return sorted UI options grouped by another column."""
    return {
        str(group): sorted(values.dropna().astype(str).unique().tolist())
        for group, values in dataframe.groupby(group_column)[value_column]
    }


def main() -> None:
    dataframe = prepare_data()
    inputs = dataframe[MODEL_FEATURES].copy()
    target = dataframe[TARGET].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        inputs,
        target,
        test_size=0.20,
        random_state=42,
    )

    pipeline = build_tuned_pipeline()
    pipeline.fit(x_train, y_train)

    train_predictions = pipeline.predict(x_train)
    test_predictions = pipeline.predict(x_test)

    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_rmse = float(np.sqrt(test_mse))
    test_r2 = r2_score(y_test, test_predictions)

    joblib.dump(pipeline, MODEL_PATH, compress=3)

    recent_year = int(dataframe["sale_year"].max())
    recent_data = dataframe[dataframe["sale_year"] == recent_year]
    recent_month = int(recent_data["sale_month"].max())
    market_medians = (
        recent_data.groupby(["town", "flat_type"])[TARGET]
        .median()
        .round(2)
    )

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(x_train)),
        "testing_rows": int(len(x_test)),
        "dataset_rows": int(len(dataframe)),
        "data_year_min": int(dataframe["sale_year"].min()),
        "data_year_max": recent_year,
        "data_month_max": recent_month,
        "best_parameters": BEST_PARAMETERS,
        "metrics": {
            "train_mae": float(train_mae),
            "test_mae": float(test_mae),
            "test_mse": float(test_mse),
            "test_rmse": test_rmse,
            "test_r2": float(test_r2),
        },
        "towns": sorted(dataframe["town"].unique().tolist()),
        "streets_by_town": nested_options(
            dataframe, "town", "street_name"
        ),
        "flat_types": sorted(dataframe["flat_type"].unique().tolist()),
        "flat_models_by_flat_type": nested_options(
            dataframe, "flat_type", "flat_model"
        ),
        "storey_ranges": sort_storey_ranges(
            dataframe["storey_range"].unique().tolist()
        ),
        "floor_area_min": float(dataframe["floor_area_sqm"].min()),
        "floor_area_max": float(dataframe["floor_area_sqm"].max()),
        "lease_years_min": float(
            dataframe["remaining_lease_years"].min()
        ),
        "lease_years_max": float(
            dataframe["remaining_lease_years"].max()
        ),
        "market_medians": {
            f"{town}|{flat_type}": float(value)
            for (town, flat_type), value in market_medians.items()
        },
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved model: {MODEL_PATH.name}")
    print(f"Saved metadata: {METADATA_PATH.name}")
    print(f"Train MAE: S${train_mae:,.2f}")
    print(f"Test MAE: S${test_mae:,.2f}")
    print(f"Test RMSE: S${test_rmse:,.2f}")
    print(f"Test R2: {test_r2:.4f}")


if __name__ == "__main__":
    main()
