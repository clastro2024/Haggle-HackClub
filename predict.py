"""Train the Week 1 project-hour model and write predictions.csv in the current directory.

All supervised fits use train.csv. Models share information across project types,
then specialize using the project's category, hosting, and patterns of development.
The final weights operate on log1p(hours); only the CSV writer converts back to hours.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve, solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
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


def kernel_ridge(target, test_size, kernel_block, penalty, center=True):
    # Block construction and an in-place Cholesky solve keep the full 4,500-row
    # fit to a single float32 training matrix plus a small working block.
    count = len(target)
    gram = np.empty((count, count), dtype=np.float32, order="F")
    for start in range(0, count, 128):
        rows = slice(start, start + 128)
        gram[rows] = kernel_block(rows, False)
    gram.flat[::count + 1] += penalty
    mean = target.mean() if center else 0.0
    centered = (target - mean).astype(np.float32)
    dual = solve(gram, centered, assume_a="pos", overwrite_a=True)
    del gram
    predicted = np.empty(test_size)
    for start in range(0, test_size, 128):
        rows = slice(start, start + 128)
        predicted[rows] = mean + kernel_block(rows, True) @ dual
    return predicted, dual


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

    prediction, _ = kernel_ridge(target, len(test), kernel, penalty=0.1)
    return prediction


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


# Separate estimates let the final regressor distinguish a large initial upload
# from sustained development, or a small codebase from an asset-heavy game.
FEATURE_VIEWS = (
    ("active_commit_days", "project_span_days", "num_commits", "num_commit_authors",
     "pct_commits_late_night", "pct_commits_weekend", "pct_generic_commit_msgs",
     "avg_commit_msg_length", "num_merge_commits", "num_branches", "num_pull_requests",
     "commit_cadence", "active_share", "active_per_person"),
    ("lines_of_code", "num_source_files", "num_directories", "num_languages",
     "pct_primary_language", "num_dependencies", "max_dir_depth", "avg_file_length",
     "has_tests", "has_backend", "has_database", "primary_language", "framework", "comment_line_ratio"),
    ("churn_ratio", "lines_added_total", "lines_deleted_total", "pct_code_in_first_commit",
     "median_lines_per_commit", "max_lines_in_one_commit", "nonmerge_commits", "edits_per_author",
     "edits_per_day", "typical_edited", "outside_largest", "edited_to_source"),
    ("readme_word_count", "description_word_count", "readme_num_headings", "readme_num_images",
     "readme_num_code_blocks", "has_license", "has_docs_folder", "readme_has_install_instructions",
     "stars", "num_issues", "num_releases"),
    ("asset_size_mb", "num_image_files", "num_audio_files", "num_3d_model_files", "num_font_files",
     "project_category", "is_multiplayer", "has_custom_art", "uses_game_engine", "framework"),
    ("hosting_type", "url_is_live", "load_time_ms", "page_weight_kb", "num_js_files_loaded",
     "num_internal_pages", "has_custom_domain", "is_mobile_responsive", "author_prior_projects",
     "author_account_age_years", "is_clone_or_tutorial", "is_forked_from_template"),
)


def specialist_features(frame):
    x = frame.copy()
    x["span_surplus"] = x["active_commit_days"] - x["project_span_days"]
    x["typical_vs_biggest"] = x["median_lines_per_commit"] / (1 + x["max_lines_in_one_commit"])
    x["rewriting"] = np.log1p(x["edits"]) - np.log1p(x["precise_loc"])
    authors = x["num_commit_authors"].clip(lower=1)
    for power in (0.5, 1):
        contributors = authors**power
        for name, column in (("active", "active_commit_days"), ("loc", "precise_loc"), ("change", "edits")):
            x[f"{name}_equiv{power}"] = x[column] / contributors
    after_first = 1 - x["pct_code_in_first_commit"]
    x["returned_after_upload"] = x["active_commit_days"] * after_first
    x["nontrivial_commits"] = x["specific_messages"] * after_first
    x["remaining_lines"] = x["lines_added_total"] * after_first
    x["sampled_avg"] = x["edits"] / (1 + x["num_commits"])
    x["median_ratio"] = x["median_lines_per_commit"] / (1 + x["sampled_avg"])
    x["noncomment_lines"] = x["precise_loc"] * (1 - x["comment_line_ratio"])
    x["secondary_code"] = x["precise_loc"] * (1 - x["pct_primary_language"])
    x["readme_length_per_description"] = np.log1p(x["readme_word_count"]) - np.log1p(x["description_word_count"])
    x["authored_asset_ratio"] = np.log1p(x["asset_count"]) - np.log1p(x["num_source_files"])
    return x


def project_segment(frame):
    kind = frame["project_category"].fillna("other").to_numpy(dtype=str)
    host = frame["hosting_type"].fillna("other").to_numpy(dtype=str)
    static_site = (kind == "website") & np.isin(host, ["github-pages", "netlify"])
    return np.select(
        [host == "itch", kind == "game", static_site, kind == "website", kind == "app", kind == "tool"],
        ["itch", "game", "static", "web", "app", "tool"], default="other",
    )


def view_inputs(train, test):
    numeric = train.select_dtypes(include=np.number).columns
    cats = train.columns.difference(numeric)
    left, right = normalize_block(*numeric_basis(train[numeric], test[numeric]))
    if not len(cats):
        return left, right
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    a = encoder.fit_transform(train[cats].astype("string").fillna("?"))
    b = encoder.transform(test[cats].astype("string").fillna("?"))
    a, b = normalize_block(a, b)
    return np.column_stack([left, a]), np.column_stack([right, b])


def project_estimates(train, test, target, recording):
    responses = np.column_stack([
        target, np.eye(3)[recording], target > np.log1p(8), target > np.log1p(25),
        np.minimum(target, np.log1p(12)),
    ])
    train_predictions, test_predictions = [], []
    for columns in (*FEATURE_VIEWS, train.columns):
        a, b = view_inputs(train[list(columns)], test[list(columns)])
        a = np.column_stack([np.ones(len(a)), a])
        b = np.column_stack([np.ones(len(b)), b])
        gram = a.T @ a
        gram.flat[::len(gram) + 1] += 1.6
        chol = cho_factor(gram, lower=True)
        coef = cho_solve(chol, a.T @ responses)
        # Remove each row's own label from its fitted value before a second model
        # sees it. The ridge hat diagonal gives the leave-one-out adjustment.
        leverage = np.square(solve_triangular(chol[0], a.T, lower=True)).sum(axis=0)
        keep = np.maximum(1 - leverage, 1e-7)[:, None]
        train_predictions.append(responses + (a @ coef - responses) / keep)
        test_predictions.append(b @ coef)
    return np.column_stack(train_predictions), np.column_stack(test_predictions)


def estimate_kernel(train, test, train_group, test_group, target, shared_train, shared_test):
    scaler = StandardScaler()
    a = scaler.fit_transform(train).clip(-4, 4)
    b = scaler.transform(test).clip(-4, 4)

    def kernel(rows, testing):
        query = (b if testing else a)[rows]
        group = (test_group if testing else train_group)[rows]
        radius = cdist(query, a) / np.sqrt(a.shape[1])
        same = group[:, None] == train_group[None, :]
        return (1 + radius) * np.exp(-radius) * (1 + 0.65 * same) + 0.12 * (query @ a.T) / a.shape[1]

    # The full-view estimate already supplies an intercept. Fit the residual
    # directly, retaining its mean in the training solve.
    residual = target - shared_train
    correction, _ = kernel_ridge(residual, len(test), kernel, penalty=0.4, center=False)
    return shared_test + correction


def adapted_networks(train, test, target, recording, train_group, test_group):
    numeric = train.select_dtypes(include=np.number).columns
    cats = train.columns.difference(numeric)
    a, b = normalize_block(*numeric_basis(train[numeric], test[numeric]))
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ca = encoder.fit_transform(train[cats].astype("string").fillna("?"))
    cb = encoder.transform(test[cats].astype("string").fillna("?"))
    a = np.column_stack([9 * a, ca]).astype(np.float32)
    b = np.column_stack([9 * b, cb]).astype(np.float32)

    centered = target - target.mean()
    styles = np.eye(3)[recording]
    labels = np.column_stack([centered, styles, styles * centered[:, None],
                              target[:, None] > np.log1p([4, 15, 40])])
    means = labels.mean(axis=0)
    labels = (labels - means).astype(np.float32)

    def estimate(net, inputs):
        heads = net.predict(inputs) + means
        return target.mean() + 0.65 * heads[:, 0] + 0.35 * heads[:, 4:7].sum(axis=1)

    predictions = []
    for seed in (51, 34, 76):
        rng = np.random.default_rng(seed)

        def masked(values):
            return values * ((rng.random(values.shape) > 0.12) / 0.88).astype(np.float32)

        shared = MLPRegressor(hidden_layer_sizes=(80, 32), alpha=1.0, learning_rate_init=0.001,
                              batch_size=min(128, len(a)), random_state=seed)
        for _ in range(32):
            shared.partial_fit(masked(a), labels)
        result = estimate(shared, b)
        # Adapt independent copies. Replay other kinds of projects to regularize
        # each specialist. Iterate training groups so predictions are batch invariant.
        for group in np.unique(train_group):
            own = np.flatnonzero(train_group == group)
            other = np.flatnonzero(train_group != group)
            if len(own) < 50 or not len(other):
                continue
            expert = deepcopy(shared)
            expert.batch_size = min(expert.batch_size, len(own))
            for _ in range(18):
                sample = np.r_[own, rng.choice(other, len(own) // 3)]
                expert.partial_fit(masked(a[sample]), labels[sample])
            use = test_group == group
            if np.any(use):
                result[use] = estimate(expert, b[use])
        predictions.append(result)
    return np.mean(predictions, axis=0)


def encoded_projects(train, test, categories):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    a = encoder.fit_transform(train[categories].astype("string").fillna("?"))
    b = encoder.transform(test[categories].astype("string").fillna("?"))
    return a, b


def segment_boost(train, test, target, train_group, test_group):
    options = dict(max_iter=165, max_leaf_nodes=9, learning_rate=0.045,
                   l2_regularization=6, early_stopping=False, random_state=42)
    predicted = np.empty(len(test))
    global_model = None
    for group in np.unique(test_group):
        own = train_group == group
        rows = test_group == group
        if own.sum() >= 70:
            model = HistGradientBoostingRegressor(**options).fit(train[own], target[own])
        else:
            if global_model is None:
                global_model = HistGradientBoostingRegressor(**options).fit(train, target)
            model = global_model
        predicted[rows] = model.predict(test[rows])
    return predicted


def duration_hurdle(train, test, target, shared_train, shared_test):
    """Predict the conditional mean on each side of 20 hours, then integrate over both."""
    options = dict(max_iter=145, min_samples_leaf=22, max_leaf_nodes=9, learning_rate=0.045,
                   l2_regularization=6, early_stopping=False, random_state=42)
    long = target > np.log1p(20)
    residual = target - shared_train
    if long.all() or not long.any():
        return shared_test + HistGradientBoostingRegressor(**options).fit(train, residual).predict(test)
    gate = HistGradientBoostingClassifier(**options).fit(train, long)
    probabilities = gate.predict_proba(test)
    prediction = shared_test.copy()
    for j, group in enumerate(gate.classes_):
        selected = long == group
        regressor = HistGradientBoostingRegressor(**options).fit(train[selected], residual[selected])
        prediction += probabilities[:, j] * regressor.predict(test)
    return prediction


def predict_models(train, test, hours, log_hours=None):
    target = np.log1p(np.asarray(hours))
    recorded = np.asarray(log_hours) if log_hours is not None else np.log(np.maximum(hours, 1e-10))
    recording = recording_groups(recorded)
    global_numeric = list(train.select_dtypes(include=np.number)) + [
        "commits_per_day", "active_share", "added_after_first", "edits", "edits_per_author",
        "edits_per_day", "nonmerge_commits", "asset_count",
    ]
    base, query = add_features(train), add_features(test)
    numeric = base.select_dtypes(include=np.number).columns
    cats = [c for c in base if c not in numeric]
    a, b = encoded_projects(base, query, cats)
    fa, fb = project_frequencies(base, query)
    frequency = predict_experts(np.column_stack([base[numeric], fa, a]),
                               np.column_stack([query[numeric], fb, b]), target, recording)
    radial = predict_similarity(base[numeric], query[numeric], a, b, len(cats), target)
    global_fit = predict_global(base, query, pd.Index(global_numeric), sorted(cats), target)

    train_group, test_group = project_segment(base), project_segment(query)
    a, b = encoded_projects(base, query, sorted(cats))
    local = segment_boost(np.column_stack([base[numeric], a]), np.column_stack([query[numeric], b]),
                          target, train_group, test_group)

    projects, incoming = specialist_features(base), specialist_features(query)
    estimates, predicted = project_estimates(projects, incoming, target, recording)
    # Each view emits seven outputs. The first output in the last block is the
    # all-features estimate of log1p(hours).
    shared_train, shared_test = estimates[:, -7], predicted[:, -7]
    similarity = estimate_kernel(estimates, predicted, train_group, test_group, target, shared_train, shared_test)
    view_experts = predict_experts(estimates, predicted, target, recording)
    full_numeric = projects.select_dtypes(include=np.number).columns
    hurdle = duration_hurdle(np.column_stack([projects[full_numeric], a, estimates]),
                             np.column_stack([incoming[full_numeric], b, predicted]),
                             target, shared_train, shared_test)
    networks = adapted_networks(projects, incoming, target, recording, train_group, test_group)
    return (0.10 * global_fit + 0.09 * radial + 0.13 * frequency + 0.20 * similarity
            + 0.17 * view_experts + 0.08 * local + 0.10 * hurdle + 0.13 * networks)


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

