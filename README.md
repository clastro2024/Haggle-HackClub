# Haggle HackClub Models (by Clastron)

This is my try with the [Haggle Competition by HackClub](https://haggle.hackclub.com/)

**Currently:** _Week 1_

**Note:** This is my first ever actual HackClub competition participation and using Hackatime and also really wanted to get a submission as fast as possible, so the code and structuring are not really the highest quality right now :/ Soooo, please better not "haggle" over this

## Repository

- `predict.py` - Main Entrypoint
- `/archives` - n0t 10st medi4 f0r sure
- `/Week 1` - Week 1 Files:
  - `/Week 1/main.ipynb` - Main full organized pipeline
  - `/Week 1/proto.ipynb` - Just prototyping and slacking code
  - `/Week 1/train.csv` - Train Data File

## Week 1

**Currently**, I just prototyped a simple Random Forest Regressor tuned with Grid Search CV, as the model.

_Best parameters found:_
`{'ccp_alpha': 0.0, 'max_depth': 12, 'min_samples_leaf': 10, 'min_samples_split': 10, 'n_estimators': 200}`

Really didn't do much EDA yet, but the main processing was:

- Separate input & output (last 3 columns) then training & validation (~10%)
- Dropped `project_id` column
- Parsed `author_account_age_years` & `` columns in min, mid and max
- Put One-Hot Encoders on all the discrete string columns

Next Steps:

- EDA (just for the show)
- Organizing and Documenting everything
- Trying out better and more trained models (well, main stuff)
