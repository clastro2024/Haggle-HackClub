import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_log_error

a = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

b = a.drop(columns=["project_id", "hours_spent", "log_hours", "hours_bucket"])
c = a[["hours_spent", "log_hours", "hours_bucket"]]

bounds = b["author_account_age_years"].str.extract(r"^\s*(<=|>=|<|>)?\s*(\d+\.?\d*)\s*(?:-\s*(\d+\.?\d*))?\s*\+?\s*$")
n1, n2 = bounds[1].astype(float), bounds[2].astype(float)

b["author_account_age_years_min"] = np.where(bounds[0].isin(["<", "<="]), 0.0, n1)
b["author_account_age_years_max"] = np.where(bounds[0].isin(["<", "<="]), n1, n2.fillna(n1))
b["author_account_age_years_mid"] = (b["author_account_age_years_min"] + b["author_account_age_years_max"]) / 2
b.drop(columns=["author_account_age_years"], inplace=True)

bounds = b["author_prior_projects"].str.extract(r"^\s*(<=|>=|<|>)?\s*(\d+\.?\d*)\s*(?:-\s*(\d+\.?\d*))?\s*\+?\s*$")
n1, n2 = bounds[1].astype(float), bounds[2].astype(float)

b["author_prior_projects_min"] = np.where(bounds[0].isin(["<", "<="]), 0.0, n1)
b["author_prior_projects_max"] = np.where(bounds[0].isin(["<", "<="]), n1, n2.fillna(n1))
b["author_prior_projects_mid"] = (b["author_prior_projects_min"] + b["author_prior_projects_max"]) / 2
b.drop(columns=["author_prior_projects"], inplace=True)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.set_output(transform="pandas")

text_cols = ["commit_cadence", "primary_language", "framework", "uses_game_engine", "has_tests", "has_ci_config", "has_dockerfile", "has_backend", "has_database", "has_custom_art", "readme_has_install_instructions", "has_license", "has_docs_folder", "url_is_live", "hosting_type", "is_mobile_responsive", "has_custom_domain", "project_category", "is_multiplayer", "mentions_ai_api", "is_clone_or_tutorial", "is_forked_from_template"]
num_cols = list(set(b.columns).difference(set(text_cols)))

# 3. Fit + transform on train, ONLY transform on test
b_ohe = ohe.fit_transform(b[text_cols])

# 4. Rejoin encoded text columns with numeric columns
b_final = pd.concat([b[num_cols], b_ohe], axis=1)

# forest = RandomForestRegressor(max_depth=12, min_samples_leaf=10, min_samples_split=10, n_estimators=200)
# forest.fit(b_final, c["hours_spent"])

forest = joblib.load("best_tree.pkl")


bounds = test["author_account_age_years"].str.extract(r"^\s*(<=|>=|<|>)?\s*(\d+\.?\d*)\s*(?:-\s*(\d+\.?\d*))?\s*\+?\s*$")
n1, n2 = bounds[1].astype(float), bounds[2].astype(float)

test["author_account_age_years_min"] = np.where(bounds[0].isin(["<", "<="]), 0.0, n1)
test["author_account_age_years_max"] = np.where(bounds[0].isin(["<", "<="]), n1, n2.fillna(n1))
test["author_account_age_years_mid"] = (test["author_account_age_years_min"] + test["author_account_age_years_max"]) / 2
test.drop(columns=["author_account_age_years"], inplace=True)

bounds = test["author_prior_projects"].str.extract(r"^\s*(<=|>=|<|>)?\s*(\d+\.?\d*)\s*(?:-\s*(\d+\.?\d*))?\s*\+?\s*$")
n1, n2 = bounds[1].astype(float), bounds[2].astype(float)

test["author_prior_projects_min"] = np.where(bounds[0].isin(["<", "<="]), 0.0, n1)
test["author_prior_projects_max"] = np.where(bounds[0].isin(["<", "<="]), n1, n2.fillna(n1))
test["author_prior_projects_mid"] = (test["author_prior_projects_min"] + test["author_prior_projects_max"]) / 2
test.drop(columns=["author_prior_projects"], inplace=True)

text_cols = ["commit_cadence", "primary_language", "framework", "uses_game_engine", "has_tests", "has_ci_config", "has_dockerfile", "has_backend", "has_database", "has_custom_art", "readme_has_install_instructions", "has_license", "has_docs_folder", "url_is_live", "hosting_type", "is_mobile_responsive", "has_custom_domain", "project_category", "is_multiplayer", "mentions_ai_api", "is_clone_or_tutorial", "is_forked_from_template"]
num_cols = list(set(test.columns).difference(set(text_cols)))

# 3. Fit + transform on train, ONLY transform on test
test_ohe = ohe.transform(test[text_cols])

# 4. Rejoin encoded text columns with numeric columns
test_final = pd.concat([test[num_cols], test_ohe], axis=1)


pred = forest.predict(test_final.drop(columns=["project_id"]))

result_df = pd.DataFrame({
    'project_id': test['project_id'],
    'hours_spent': pred,
})

result_df.to_csv('predictions.csv', index=False)

