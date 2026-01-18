import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta

# ----------------------------
# Load model and data
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("traffic.csv")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df

model = load_model()
df = load_data()

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Traffic Prediction", layout="centered")

st.title("🚦 Traffic Flow Prediction")
st.write("Predict traffic using junction and time. Features are generated automatically.")

# User Inputs
junction = st.selectbox("🔀 Select Junction", sorted(df["Junction"].unique()))
selected_datetime = st.datetime_input("⏰ Select Date & Time", datetime.now())

# ----------------------------
# Feature Engineering
# ----------------------------

def generate_features(junction, dt):

    data = df[df["Junction"] == junction].sort_values("DateTime")

    def get_vehicles_at(time):
        row = data[data["DateTime"] == time]
        return row["Vehicles"].values[0] if not row.empty else data["Vehicles"].mean()

    lag_1 = get_vehicles_at(dt - timedelta(hours=1))
    lag_24 = get_vehicles_at(dt - timedelta(hours=24))
    lag_168 = get_vehicles_at(dt - timedelta(hours=168))

    last_3 = data[(data["DateTime"] < dt) & (data["DateTime"] >= dt - timedelta(hours=3))]["Vehicles"]
    last_6 = data[(data["DateTime"] < dt) & (data["DateTime"] >= dt - timedelta(hours=6))]["Vehicles"]
    last_24 = data[(data["DateTime"] < dt) & (data["DateTime"] >= dt - timedelta(hours=24))]["Vehicles"]

    features = {
        # EXACT training features
        "Junction": junction,
        "ID": data["ID"].iloc[0],
        "Hour": dt.hour,
        "Day": dt.day,
        "Month": dt.month,
        "Weekday": dt.weekday(),
        "Is_Weekend": 1 if dt.weekday() >= 5 else 0,
        "Lag_1": lag_1,
        "Lag_24": lag_24,
        "Lag_168": lag_168,
        "Roll_Mean_3": last_3.mean(),
        "Roll_Mean_6": last_6.mean(),
        "Roll_Mean_24": last_24.mean()
    }

    return pd.DataFrame([features])

def get_traffic_level(predicted_vehicles):
    low_threshold = df["Vehicles"].quantile(0.70)
    high_threshold = df["Vehicles"].quantile(0.90)
    print(low_threshold,high_threshold)
    if predicted_vehicles > high_threshold:
        return "🔴 High"
    elif predicted_vehicles > low_threshold:
        return "🟡 Medium"
    else:
        return "🟢 Low"



# ----------------------------
# Prediction
# ----------------------------
if st.button("🚀 Predict Traffic"):
    feature_df = generate_features(junction, selected_datetime)
    prediction = model.predict(feature_df)[0]

    traffic_level = get_traffic_level(prediction)

    st.success(
            f"🚗 Predicted Traffic Volume: **{int(prediction)} vehicles**\n\n"
            f"🚦 Traffic Level: **{traffic_level}**"
        )    

    with st.expander("🔍 See generated features"):
        st.dataframe(feature_df)
