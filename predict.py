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


pred = forest.predict(test_final.drop(columns=["project_id"])[['pct_code_in_first_commit',
 'page_weight_kb',
 'num_directories',
 'author_prior_projects_mid',
 'project_span_days',
 'author_prior_projects_max',
 'pct_primary_language',
 'readme_word_count',
 'avg_file_length',
 'num_internal_pages',
 'readme_num_images',
 'num_issues',
 'median_lines_per_commit',
 'load_time_ms',
 'pct_commits_weekend',
 'num_branches',
 'num_test_files',
 'num_commits',
 'pct_generic_commit_msgs',
 'author_prior_projects_min',
 'num_releases',
 'num_dependencies',
 'forks',
 'avg_commit_msg_length',
 'comment_line_ratio',
 'author_account_age_years_max',
 'num_pull_requests',
 'max_lines_in_one_commit',
 'num_commit_authors',
 'lines_of_code',
 'active_commit_days',
 'pct_commits_late_night',
 'max_dir_depth',
 'num_font_files',
 'num_3d_model_files',
 'num_audio_files',
 'asset_size_mb',
 'author_account_age_years_min',
 'readme_num_headings',
 'description_word_count',
 'num_merge_commits',
 'stars',
 'num_languages',
 'num_js_files_loaded',
 'churn_ratio',
 'author_account_age_years_mid',
 'lines_deleted_total',
 'readme_num_code_blocks',
 'num_image_files',
 'lines_added_total',
 'num_source_files',
 'commit_cadence_daily',
 'commit_cadence_frequent',
 'commit_cadence_sporadic',
 'primary_language_Astro',
 'primary_language_C',
 'primary_language_C#',
 'primary_language_C++',
 'primary_language_CSS',
 'primary_language_Dart',
 'primary_language_GDScript',
 'primary_language_Go',
 'primary_language_HTML',
 'primary_language_Java',
 'primary_language_JavaScript',
 'primary_language_Kotlin',
 'primary_language_Lua',
 'primary_language_Python',
 "primary_language_Ren'Py",
 'primary_language_Rust',
 'primary_language_Shader',
 'primary_language_Svelte',
 'primary_language_Swift',
 'primary_language_TypeScript',
 'primary_language_none',
 'primary_language_other',
 'framework_astro',
 'framework_discord-bot',
 'framework_express',
 'framework_fastapi',
 'framework_flask',
 'framework_gamemaker',
 'framework_godot',
 'framework_nextjs',
 'framework_none',
 'framework_other',
 'framework_pygame',
 'framework_react',
 'framework_react-native',
 'framework_renpy',
 'framework_slack-bot',
 'framework_sveltekit',
 'framework_threejs',
 'framework_unity',
 'uses_game_engine_False',
 'uses_game_engine_True',
 'has_tests_False',
 'has_tests_True',
 'has_ci_config_False',
 'has_ci_config_True',
 'has_dockerfile_False',
 'has_dockerfile_True',
 'has_backend_False',
 'has_backend_True',
 'has_database_False',
 'has_database_True',
 'has_custom_art_False',
 'has_custom_art_True',
 'has_custom_art_nan',
 'readme_has_install_instructions_False',
 'readme_has_install_instructions_True',
 'has_license_False',
 'has_license_True',
 'has_docs_folder_False',
 'has_docs_folder_True',
 'url_is_live_False',
 'url_is_live_True',
 'hosting_type_chat-invite',
 'hosting_type_cloudflare-pages',
 'hosting_type_custom',
 'hosting_type_github-pages',
 'hosting_type_github-repo',
 'hosting_type_hack-club',
 'hosting_type_huggingface',
 'hosting_type_itch',
 'hosting_type_mod-platform',
 'hosting_type_netlify',
 'hosting_type_none',
 'hosting_type_other',
 'hosting_type_package-registry',
 'hosting_type_render',
 'hosting_type_streamlit',
 'hosting_type_vercel',
 'hosting_type_video',
 'is_mobile_responsive_False',
 'is_mobile_responsive_True',
 'is_mobile_responsive_nan',
 'has_custom_domain_False',
 'has_custom_domain_True',
 'project_category_app',
 'project_category_bot',
 'project_category_game',
 'project_category_other',
 'project_category_tool',
 'project_category_website',
 'project_category_nan',
 'is_multiplayer_False',
 'is_multiplayer_True',
 'is_multiplayer_nan',
 'mentions_ai_api_False',
 'mentions_ai_api_True',
 'mentions_ai_api_nan',
 'is_clone_or_tutorial_False',
 'is_clone_or_tutorial_True',
 'is_clone_or_tutorial_nan',
 'is_forked_from_template_False',
 'is_forked_from_template_True']])

result_df = pd.DataFrame({
    'project_id': test['project_id'],
    'hours_spent': pred,
})

result_df.to_csv('predictions.csv', index=False)

