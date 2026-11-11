# Titanic Survival Predictor

Predicts survival probability on the Titanic dataset.

- Model: RandomForestClassifier (200 trees), ~79% test accuracy
- Data: seaborn's built-in `titanic` dataset (891 rows)
- Features: pclass, age, sibsp, parch, fare, alone, sex_male, embarked_Q, embarked_S

## Run it
1. `python -m pip install -r requirements.txt`
2. `python train.py`
3. `python -m streamlit run app.py`
