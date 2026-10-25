"""
Olympic Participation Predictor - Streamlit web app
Loads the model trained in the Colab notebook (olympic_participation_model.pkl)
and predicts how many Olympic Games an athlete will take part in.
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Olympic Participation Predictor", page_icon="🏅", layout="centered")

MODEL_PATH = Path(__file__).parent / "olympic_participation_model.pkl"


@st.cache_resource
def load_bundle():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


bundle = load_bundle()
model = bundle["model"]

st.title("🏅 Olympic Participation Predictor")
st.write(
    f"This app predicts **how many Olympic Games** an athlete will take part in. "
    f"Model used: **{bundle['model_name']}** (best of 5 algorithms in 10-fold cross-validation, "
    f"mean RMSE about {bundle['cv_rmse']:.2f})."
)

st.subheader("Athlete details")
col1, col2 = st.columns(2)

with col1:
    season = st.selectbox("Season of first Games", ["Summer", "Winter"])
    first_year = st.number_input("Year of first Games", min_value=1896, max_value=2024, value=2008, step=1)
    birth_year = st.number_input("Birth year", min_value=1850, max_value=2010, value=1990, step=1)
    has_bio = st.selectbox("Has a biography on the Olympics website?", ["Yes", "No"])

with col2:
    gold = st.number_input("Gold medals", min_value=0, max_value=30, value=0)
    silver = st.number_input("Silver medals", min_value=0, max_value=30, value=0)
    bronze = st.number_input("Bronze medals", min_value=0, max_value=30, value=0)

if st.button("Predict", type="primary"):
    age_first = first_year - birth_year
    if age_first < 8 or age_first > 80:
        st.warning(f"Age at first Games would be {age_first}. Please check the years.")
    else:
        row = pd.DataFrame([{
            "season": season,
            "first_game_year": first_year,
            "birth_year": birth_year,
            "age_at_first_game": age_first,
            "gold": gold,
            "silver": silver,
            "bronze": bronze,
            "has_bio": 1 if has_bio == "Yes" else 0,
        }])[bundle["feature_columns"]]

        pred = max(float(model.predict(row)[0]), 1.0)
        st.success(f"Predicted Games participations: **{pred:.2f}** (about {round(pred)})")
        st.caption("Most athletes take part only once, so predictions are usually close to 1 or 2. This is an estimate, not a fact.")
