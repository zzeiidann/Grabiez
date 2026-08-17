from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "grabcar_deployment_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "grabcar_deployment_model.json"

CATEGORICAL_FEATURES = ["type", "service_tier_id"]
NUMERIC_FEATURES = [
    "distance_mean",
    "humidity",
    "rain",
    "temp",
    "wind",
    "clouds",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    timestamp = pd.to_datetime(result["timestamp"])
    hour = timestamp.dt.hour + timestamp.dt.minute / 60
    day_of_week = timestamp.dt.dayofweek
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    result["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    result["is_weekend"] = day_of_week.isin([5, 6]).astype("int8")
    return result


def chronological_splits(frame: pd.DataFrame, n_splits: int = 3):
    timestamps = np.array(sorted(frame["timestamp"].unique()))
    blocks = np.array_split(timestamps, n_splits + 1)
    for fold in range(n_splits):
        train_times = np.concatenate(blocks[: fold + 1])
        valid_times = blocks[fold + 1]
        yield (
            np.flatnonzero(frame["timestamp"].isin(train_times)),
            np.flatnonzero(frame["timestamp"].isin(valid_times)),
        )


def make_pipeline(alpha: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "interactions",
                PolynomialFeatures(
                    degree=2,
                    interaction_only=True,
                    include_bias=False,
                ),
            ),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["timestamp"])
    mapping = pd.read_csv(ROOT / "artifacts" / "type_cluster_mapping.csv")
    train = train.merge(
        mapping[["type", "service_tier_id"]],
        on="type",
        how="left",
        validate="many_to_one",
    )
    train = add_time_features(train).sort_values("timestamp").reset_index(drop=True)
    if train["service_tier_id"].isna().any():
        raise ValueError("Some train types do not have a service-tier mapping.")

    alphas = np.logspace(-3, 2, 16)
    scores: dict[float, float] = {}
    for alpha in alphas:
        fold_scores = []
        for train_idx, valid_idx in chronological_splits(train):
            pipeline = make_pipeline(float(alpha))
            pipeline.fit(train.loc[train_idx, FEATURES], train.loc[train_idx, "price_mean"])
            prediction = pipeline.predict(train.loc[valid_idx, FEATURES])
            fold_scores.append(
                mean_squared_error(train.loc[valid_idx, "price_mean"], prediction) ** 0.5
            )
        scores[float(alpha)] = float(np.mean(fold_scores))

    best_alpha = min(scores, key=scores.get)
    model = make_pipeline(best_alpha)
    model.fit(train[FEATURES], train["price_mean"])

    type_counts = train.groupby("type").size().astype(float)
    type_weights = {
        str(tier): {
            str(raw_type): float(type_counts.loc[raw_type])
            for raw_type in group["type"].tolist()
        }
        for tier, group in mapping.groupby("service_tier_id")
    }
    tier_names = (
        mapping[["service_tier_id", "service_tier"]]
        .drop_duplicates()
        .set_index("service_tier_id")["service_tier"]
        .to_dict()
    )

    artifact = {
        "model": model,
        "features": FEATURES,
        "type_weights": type_weights,
        "tier_names": {str(key): value for key, value in tier_names.items()},
        "price_scale_idr": 1000.0,
        "best_alpha": best_alpha,
        "cv_rmse": scores[best_alpha],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "features": FEATURES,
                "best_alpha": best_alpha,
                "chronological_cv_rmse": scores[best_alpha],
                "alpha_scores": scores,
                "price_scale_idr": artifact["price_scale_idr"],
                "notes": (
                    "Production feature contract excludes API-call and surge aggregates. "
                    "Raw competition type is marginalized within each service tier at inference."
                ),
            },
            indent=2,
        )
    )
    print(f"Saved {ARTIFACT_PATH}")
    print(f"Best alpha: {best_alpha:.6g}")
    print(f"Chronological CV RMSE: {scores[best_alpha]:.6f}")


if __name__ == "__main__":
    main()
