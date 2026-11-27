"""Train on train.csv and write Week 1 hour predictions for test.csv."""

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits


LABELS = ["hours_spent", "log_hours", "hours_bucket"]
ACTIVITY = ["active_commit_days", "num_commits", "lines_of_code", "lines_added_total",
            "lines_deleted_total", "num_source_files", "project_span_days", "readme_word_count"]
FREQUENCY_KEYS = [
    ("num_commits", "pct_commits_weekend", "avg_commit_msg_length", "lines_added_total"),
    ("num_source_files", "avg_file_length", "comment_line_ratio"),
    ("readme_word_count", "readme_num_headings", "readme_num_images", "readme_num_code_blocks"),
    ("num_directories", "num_source_files", "num_dependencies", "primary_language"),
    ("num_image_files", "num_audio_files", "num_font_files", "asset_size_mb"),
    ("hosting_type", "page_weight_kb", "num_internal_pages", "num_js_files_loaded"),
    ("description_word_count",), ("readme_word_count",), ("project_span_days",),
    ("framework", "primary_language", "hosting_type", "project_category"),
    ("median_lines_per_commit",), ("avg_commit_msg_length",),
]


def add_features(frame):
    x = frame.copy()
    days = x["active_commit_days"].clip(lower=1)
    authors = x["num_commit_authors"].clip(lower=1)
    edits = x["lines_added_total"] + x["lines_deleted_total"]
    x["commits_per_day"] = x["num_commits"] / days
    x["active_share"] = days / (1 + x["project_span_days"])
    x["added_after_first"] = x["lines_added_total"] * (1 - x["pct_code_in_first_commit"])
    x["edits"] = edits
    x["edits_per_author"] = edits / authors
    x["edits_per_day"] = edits / days
    x["nonmerge_commits"] = x["num_commits"] - x["num_merge_commits"]
    x["asset_count"] = x[["num_image_files", "num_audio_files", "num_3d_model_files",
                          "num_font_files"]].sum(axis=1, min_count=1)
    x["precise_loc"] = x["num_source_files"] * x["avg_file_length"]
    x["authored_changes"] = x["nonmerge_commits"] / authors
    x["active_per_person"] = days / authors
    x["typical_edited"] = x["median_lines_per_commit"] * x["num_commits"]
    x["weekend_commits"] = x["num_commits"] * x["pct_commits_weekend"]
    x["night_commits"] = x["num_commits"] * x["pct_commits_late_night"]
    x["specific_messages"] = x["num_commits"] * (1 - x["pct_generic_commit_msgs"])
    x["outside_largest"] = (edits - x["max_lines_in_one_commit"]).clip(lower=0)
    x["lifetime_per_person"] = x["project_span_days"] / (1 + authors)
    x["document_density"] = x["readme_word_count"] / (1 + x["num_source_files"])
    x["activity_per_file"] = x["num_commits"] / (1 + x["num_source_files"])
    x["edited_to_source"] = edits / (1 + x["precise_loc"])
    x["words_per_heading"] = x["readme_word_count"] / (1 + x["readme_num_headings"])
    x["repo_age_per_active"] = (1 + x["project_span_days"]) / days
    return x


def project_frequencies(train, test):
    """Count similar measurements in the training set, excluding a project's own row."""
    training_counts, test_counts = [], []
    for names in FREQUENCY_KEYS:
        columns = list(names)
        # Use the same hash representation if missing test values turned ints into floats.
        types = {c: "float64" if pd.api.types.is_numeric_dtype(train[c]) else "string" for c in columns}
        left = pd.util.hash_pandas_object(train[columns].astype(types), index=False)
        right = pd.util.hash_pandas_object(test[columns].astype(types), index=False)
        counts = left.value_counts()
        training_counts.append(np.log1p(left.map(counts) - 1))
        test_counts.append(np.log1p(right.map(counts).fillna(0)))
    return np.column_stack(training_counts), np.column_stack(test_counts)


def recording_groups(log_hours):
    # log_hours is stored to four decimal places. Allow for its rounding error
    # when checking whether the original measurement lies on a half- or tenth-hour grid.
    hours = np.exp(log_hours)
    def on_grid(units):
        scaled = units * hours
        return np.abs(scaled - np.rint(scaled)) <= 0.000052 * scaled + 1e-7
    return np.select([on_grid(2), on_grid(10)], [0, 1], default=2)


def predict_experts(train, test, target, groups):
    settings = dict(max_iter=150, learning_rate=0.045, max_leaf_nodes=9, min_samples_leaf=25,
                    l2_regularization=5, early_stopping=False, random_state=42)
    classes = np.unique(groups)
    if len(classes) == 1:
        return HistGradientBoostingRegressor(**settings).fit(train, target).predict(test)
    gate = HistGradientBoostingClassifier(**settings).fit(train, groups)
    probabilities = gate.predict_proba(test)
    prediction = np.zeros(len(test))
    for column, group in enumerate(gate.classes_):
        mask = groups == group
        expert = HistGradientBoostingRegressor(**settings).fit(train[mask], target[mask])
        prediction += probabilities[:, column] * expert.predict(test)
    return prediction


def soft_thresholds(values, knots, width):
    valid = np.isfinite(values)
    logs = np.log1p(np.where(valid, values, 0).clip(min=0))
    curves = expit((logs[:, None] - knots) / width)
    return np.column_stack([curves * valid[:, None], valid]) / np.sqrt(len(knots))


def numeric_basis(train, test):
    training_blocks, test_blocks = [], []
    for c in train:
        observed = train[c].dropna().to_numpy()
        if not len(observed):
            continue
        logs = np.log1p(observed.clip(min=0))
        knots = np.unique(np.quantile(logs, np.arange(0.05, 0.96, 0.1)))
        width = max(0.01, 0.075 * (np.quantile(logs, 0.9) - np.quantile(logs, 0.1)))
        training_blocks.append(soft_thresholds(train[c].to_numpy(), knots, width))
        test_blocks.append(soft_thresholds(test[c].to_numpy(), knots, width))
    return np.column_stack(training_blocks), np.column_stack(test_blocks)


def normalize_block(train, test):
    center = train.mean(axis=0)
    train = train - center
    scale = max(np.sqrt(np.square(train).sum(axis=1).mean()), 1e-8)
    return train / scale, (test - center) / scale


def kernel_ridge(target, test_size, kernel_block, penalty):
    # Block construction and an in-place Cholesky solve keep the full 4,500-row
    # fit to a single float32 training matrix plus a small working block.
    count = len(target)
    gram = np.empty((count, count), dtype=np.float32, order="F")
    for start in range(0, count, 128):
        rows = slice(start, start + 128)
        gram[rows] = kernel_block(rows, False)
    gram.flat[::count + 1] += penalty
    centered = (target - target.mean()).astype(np.float32)
    dual = solve(gram, centered, assume_a="pos", overwrite_a=True)
    del gram
    predicted = np.empty(test_size)
    for start in range(0, test_size, 128):
        rows = slice(start, start + 128)
        predicted[rows] = target.mean() + kernel_block(rows, True) @ dual
    return predicted, dual


def predict_interactions(num_train, num_test, cat_train, cat_test, target):
    nums, nums_test = normalize_block(*numeric_basis(num_train, num_test))
    cats, cats_test = normalize_block(cat_train, cat_test)
    def kernel(rows, testing):
        n = (nums_test if testing else nums)[rows] @ nums.T
        c = (cats_test if testing else cats)[rows] @ cats.T
        return 2 * n + c + 0.8 * n * c + 0.3 * n**2 + 0.15 * c**2 + 1
    predicted, _ = kernel_ridge(target, len(num_test), kernel, penalty=4)
    return predicted


def predict_similarity(num_train, num_test, cat_train, cat_test, num_categories, target):
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = StandardScaler()
    numeric = scaler.fit_transform(np.log1p(imputer.fit_transform(num_train).clip(min=0))).clip(-4, 4)
    test_numeric = scaler.transform(np.log1p(imputer.transform(num_test).clip(min=0))).clip(-4, 4)
    scale = np.sqrt(numeric.shape[1])
    train = np.column_stack([numeric / scale, cat_train / np.sqrt(num_categories)])
    test = np.column_stack([test_numeric / scale, cat_test / np.sqrt(num_categories)])

    def kernel(rows, testing):
        radius = cdist((test if testing else train)[rows], train)
        return (1 + radius) * np.exp(-radius)

    first, dual = kernel_ridge(target, len(test), kernel, penalty=0.1)
    # Learn a distance metric from the first fit's gradients. Keep a diagonal
    # component so weak directions still contribute to the second fit.
    gradients = np.empty_like(train)
    for start in range(0, len(train), 128):
        rows = slice(start, start + 128)
        weights = np.exp(-cdist(train[rows], train)) * dual
        gradients[rows] = weights @ train - weights.sum(axis=1)[:, None] * train[rows]
    values, directions = np.linalg.eigh(gradients.T @ gradients / len(train))
    strength = np.sqrt(values.clip(min=0))
    normalizer = max(1e-10, np.var(train @ directions, axis=0) @ strength)
    transform = directions * np.sqrt(strength / normalizer + 0.12)
    train, test = train @ transform, test @ transform
    second, _ = kernel_ridge(target, len(test), kernel, penalty=0.1)
    return first, second


def predict_global(train, test, numeric, categorical, target):
    """Fit the original three global regressors on their original feature set."""
    num_train, num_test = train[numeric], test[numeric]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cats = encoder.fit_transform(train[categorical].astype("string").fillna("missing"))
    test_cats = encoder.transform(test[categorical].astype("string").fillna("missing"))
    boost = HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=12, min_samples_leaf=25,
                                          l2_regularization=4, learning_rate=0.035,
                                          early_stopping=False, random_state=42)
    boost.fit(np.column_stack([num_train, cats]), target)
    boosted = boost.predict(np.column_stack([num_test, test_cats]))

    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    logs = np.log1p(imputer.fit_transform(num_train).clip(min=0))
    test_logs = np.log1p(imputer.transform(num_test).clip(min=0))
    varying = np.ptp(logs, axis=0) > 0
    splines = SplineTransformer(n_knots=5, degree=2, include_bias=False)
    ridge = Ridge(alpha=12).fit(np.column_stack([splines.fit_transform(logs[:, varying]), cats]), target)
    smooth = ridge.predict(np.column_stack([splines.transform(test_logs[:, varying]), test_cats]))

    scaler = StandardScaler()
    active = numeric.get_indexer(ACTIVITY)
    def coordinates(z, category):
        return np.column_stack([z / np.sqrt(z.shape[1]), z[:, active] / np.sqrt(len(active)),
                                category / np.sqrt(len(categorical))])
    left = coordinates(scaler.fit_transform(logs).clip(-4, 4), cats)
    right = coordinates(scaler.transform(test_logs).clip(-4, 4), test_cats)
    def kernel(rows, testing):
        r = cdist((right if testing else left)[rows], left)
        return (1 + r) * np.exp(-r)
    similar, _ = kernel_ridge(target, len(test), kernel, penalty=0.25)
    return 0.325 * boosted + 0.325 * smooth + 0.35 * similar


def predict_models(train, test, hours, log_hours=None):
    target = np.log1p(np.asarray(hours))
    groups = recording_groups(np.asarray(log_hours) if log_hours is not None
                              else np.log(np.maximum(hours, 1e-10)))
    train, test = add_features(train), add_features(test)
    numeric = train.select_dtypes(include=np.number).columns
    categorical = [c for c in train if c not in numeric]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_train = encoder.fit_transform(train[categorical].astype("string").fillna("?"))
    cat_test = encoder.transform(test[categorical].astype("string").fillna("?"))
    num_train, num_test = train[numeric], test[numeric]

    plain_train = np.column_stack([num_train, cat_train])
    plain_test = np.column_stack([num_test, cat_test])
    plain = predict_experts(plain_train, plain_test, target, groups)
    freq_train, freq_test = project_frequencies(train, test)
    frequent = predict_experts(np.column_stack([num_train, freq_train, cat_train]),
                               np.column_stack([num_test, freq_test, cat_test]), target, groups)
    interactions = predict_interactions(num_train, num_test, cat_train, cat_test, target)
    first, adapted = predict_similarity(num_train, num_test, cat_train, cat_test, len(categorical), target)
    specialists = 0.23 * plain + 0.23 * frequent + 0.34 * interactions + 0.09 * first + 0.11 * adapted
    global_columns = train.loc[:, :"asset_count"].select_dtypes(include=np.number).columns
    global_prediction = predict_global(train, test, global_columns, sorted(categorical), target)
    return 0.5 * specialists + 0.5 * global_prediction


def main():
    train = pd.read_csv("train.csv", dtype={"project_id": str})
    test = pd.read_csv("test.csv", dtype={"project_id": str})
    features = train.drop(columns=["project_id", *LABELS])
    log_predictions = np.empty(0)
    if not test.empty:
        with threadpool_limits(limits=3):
            log_predictions = predict_models(features, test[features.columns], train["hours_spent"],
                                             train["log_hours"])
    pd.DataFrame({"project_id": test["project_id"],
                  "hours_spent": np.expm1(log_predictions).clip(0, 140.7)}).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()

