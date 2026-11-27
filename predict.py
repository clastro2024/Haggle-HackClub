"""Train on train.csv and predict approved project hours for test.csv."""

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits


LABELS = ["hours_spent", "log_hours", "hours_bucket"]
ACTIVITY = ["active_commit_days", "num_commits", "lines_of_code", "lines_added_total",
            "lines_deleted_total", "num_source_files", "project_span_days", "readme_word_count"]


def add_features(frame):
    x = frame.copy()
    days = x["active_commit_days"].clip(lower=1)
    edits = x["lines_added_total"] + x["lines_deleted_total"]
    x["commits_per_day"] = x["num_commits"] / days
    x["active_share"] = days / (1 + x["project_span_days"])
    x["added_after_first"] = x["lines_added_total"] * (1 - x["pct_code_in_first_commit"])
    x["edits"] = edits
    x["edits_per_author"] = edits / x["num_commit_authors"].clip(lower=1)
    x["edits_per_day"] = edits / days
    x["nonmerge_commits"] = x["num_commits"] - x["num_merge_commits"]
    x["asset_count"] = x[["num_image_files", "num_audio_files", "num_3d_model_files",
                          "num_font_files"]].sum(axis=1, min_count=1)
    return x


def project_kernel(left, right):
    """Matérn kernel. Compute in blocks to bound peak memory on the scoring CPU."""
    matrix = np.empty((len(left), len(right)), dtype=np.float32, order="F")
    for start in range(0, len(left), 128):
        rows = slice(start, start + 128)
        distance = cdist(left[rows], right)
        matrix[rows] = (1 + distance) * np.exp(-distance)
    return matrix


def similarity_features(standardized, categories, activity, category_count):
    # Give overall numeric signals, activity signals, and categorical signals
    # their own distance scales so wide one-hot groups cannot dominate.
    return np.column_stack([
        standardized / np.sqrt(standardized.shape[1]),
        standardized[:, activity] / np.sqrt(len(activity)),
        categories / np.sqrt(category_count),
    ])


def predict_models(train, test, hours):
    target = np.log1p(hours)
    train, test = add_features(train), add_features(test)
    numeric = train.select_dtypes(include=np.number).columns
    category = train.columns.difference(numeric)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_train = encoder.fit_transform(train[category].astype("string").fillna("missing"))
    cat_test = encoder.transform(test[category].astype("string").fillna("missing"))
    num_train, num_test = train[numeric], test[numeric]

    booster = HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=12,
                                            min_samples_leaf=25, l2_regularization=4,
                                            learning_rate=0.035, early_stopping=False, random_state=42)
    booster.fit(np.column_stack([num_train, cat_train]), target)
    boost = booster.predict(np.column_stack([num_test, cat_test]))

    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    log_train = np.log1p(imputer.fit_transform(num_train).clip(min=0))
    log_test = np.log1p(imputer.transform(num_test).clip(min=0))
    varying = np.ptp(log_train, axis=0) > 0
    splines = SplineTransformer(n_knots=5, degree=2, include_bias=False)
    ridge = Ridge(alpha=12)
    basis = np.column_stack([splines.fit_transform(log_train[:, varying]), cat_train])
    basis_test = np.column_stack([splines.transform(log_test[:, varying]), cat_test])
    ridge.fit(basis, target)
    smooth = ridge.predict(basis_test)

    scale = StandardScaler()
    z_train = scale.fit_transform(log_train).clip(-4, 4)
    z_test = scale.transform(log_test).clip(-4, 4)
    activity = numeric.get_indexer(ACTIVITY)
    coordinates = similarity_features(z_train, cat_train, activity, len(category))
    test_coordinates = similarity_features(z_test, cat_test, activity, len(category))
    gram = project_kernel(coordinates, coordinates)
    gram.flat[::len(gram) + 1] += 0.25
    centered = target - target.mean()
    dual = solve(gram, centered.astype(np.float32), assume_a="pos", overwrite_a=True)
    del gram
    similarity = target.mean() + project_kernel(test_coordinates, coordinates) @ dual

    # Blend in the same log1p space used by the competition's RMSLE.
    return 0.325 * boost + 0.325 * smooth + 0.35 * similarity


def main():
    train = pd.read_csv("train.csv", dtype={"project_id": str})
    test = pd.read_csv("test.csv", dtype={"project_id": str})
    features = train.drop(columns=["project_id", *LABELS])
    if test.empty:
        log_hours = np.empty(0)
    else:
        with threadpool_limits(limits=3):
            log_hours = predict_models(features, test[features.columns], train["hours_spent"].to_numpy())
    pd.DataFrame({"project_id": test["project_id"],
                  "hours_spent": np.expm1(log_hours).clip(0, 140.7)}).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
