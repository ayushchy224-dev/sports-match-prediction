
import streamlit as st
import joblib
import pandas as pd
from datetime import date

model = joblib.load("cricket_model.pkl")

st.set_page_config(page_title="Cricket Match Predictor", page_icon="🏏", layout="centered")

st.title("🏏 Cricket Match Predictor")
st.write("Predict the likely winner using machine learning trained on historical cricket matches.")

competition = st.selectbox("Competition", ["International T20", "IPL"])

teams = [
    "Afghanistan", "Australia", "Bangladesh", "England", "India",
    "Ireland", "New Zealand", "Pakistan", "South Africa", "Sri Lanka",
    "West Indies", "Zimbabwe", "Chennai Super Kings", "Delhi Capitals",
    "Gujarat Titans", "Kolkata Knight Riders", "Lucknow Super Giants",
    "Mumbai Indians", "Punjab Kings", "Rajasthan Royals",
    "Royal Challengers Bengaluru", "Sunrisers Hyderabad"
]

team1 = st.selectbox("Team 1", teams, index=4)
team2_options = [team for team in teams if team != team1]
team2 = st.selectbox("Team 2", team2_options)

venue = st.text_input("Venue", "Arun Jaitley Stadium")
match_date = st.date_input("Match Date", value=date.today())

st.subheader("Toss Details")
toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

if st.button("Predict Winner", type="primary", use_container_width=True):
    if team1 == team2:
        st.error("Please select two different teams.")
    elif not venue.strip():
        st.error("Please enter a venue.")
    elif match_date < date.today():
        st.error("Please enter today's date or a future date.")
    else:
        row = pd.DataFrame([{
            "Competition": competition,
            "Team 1": team1,
            "Team 2": team2,
            "Venue": venue.strip(),
            "Year": match_date.year,
            "Month": match_date.month,
            "Toss Won by Team 1": int(toss_winner == team1),
            "Team 1 Chose Bat": int(
                (toss_winner == team1 and toss_decision == "bat") or
                (toss_winner == team2 and toss_decision == "field")
            )
        }])

        try:
            prediction = int(model.predict(row)[0])
            probabilities = model.predict_proba(row)[0]

            if prediction == 1:
                winner = team1
                confidence = probabilities[1] * 100
            else:
                winner = team2
                confidence = probabilities[0] * 100

            st.success(f"Predicted Winner: {winner}")
            st.metric("Estimated model probability", f"{confidence:.2f}%")
            st.caption("This is a model estimate, not a guaranteed result.")
        except Exception as error:
            st.error("Prediction failed. Check that the model and input features match.")
            st.code(str(error))
