
import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import os

# Page config
st.set_page_config(page_title="Spotify Churn Prediction", layout="wide")
st.title("🎵 Spotify Churn Prediction App")
st.markdown("---")

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load('rf_model.pkl')
    with open('feature_columns.json', 'r') as f:
        feature_cols = json.load(f)
    return model, feature_cols

model, feature_cols = load_model()
st.success("✅ Model loaded successfully!")

st.markdown("## Enter User Details")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 16, 59, 38)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])

with col2:
    country = st.selectbox("Country", ["CA", "DE", "AU", "US", "UK", "IN", "FR", "PK"])
    subscription = st.selectbox("Subscription", ["Free", "Family", "Premium", "Student"])

with col3:
    device = st.selectbox("Device", ["Desktop", "Web", "Mobile"])
    offline_listening = st.selectbox("Offline Listening", [0, 1])

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    listening_time = st.slider("Listening Time (min)", 10, 299, 154)
    songs_per_day = st.slider("Songs Played/Day", 1, 99, 50)

with col2:
    skip_rate = st.slider("Skip Rate", 0.0, 0.6, 0.3)
    ads_per_week = st.slider("Ads Listened/Week", 0, 49, 7)

if st.button("🔮 Predict Churn Risk", type="primary"):
    # Create input dataframe
    input_data = {
        'age': age,
        'listening_time': listening_time,
        'songs_played_per_day': songs_per_day,
        'skip_rate': skip_rate,
        'ads_listened_per_week': ads_per_week,
        'offline_listening': offline_listening,
        'gender_encoded': {"Female": 0, "Male": 1, "Other": 2}[gender],
        'country_encoded': {"CA": 0, "DE": 1, "AU": 2, "US": 3, "UK": 4, "IN": 5, "FR": 6, "PK": 7}[country],
        'subscription_type_encoded': {"Free": 0, "Family": 1, "Premium": 2, "Student": 3}[subscription],
        'device_type_encoded': {"Desktop": 0, "Web": 1, "Mobile": 2}[device],
        'is_premium': 1 if subscription == "Premium" else 0,
        'has_ads': 1 if ads_per_week > 0 else 0
    }

    input_df = pd.DataFrame([input_data])

    # Align features
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("### 📊 Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error(f"⚠️ **HIGH Churn Risk**")
            st.metric("Churn Probability", f"{probability:.1%}", delta="High Risk")
        else:
            st.success(f"✅ **LOW Churn Risk**")
            st.metric("Churn Probability", f"{probability:.1%}", delta="Safe")

    with col2:
        st.subheader("🎯 Retention Recommendations")
        if probability > 0.3:
            st.warning("💡 **Target for retention campaign**")
            st.info("- Offer Premium discount - Personalized playlist - Offline listening promotion")
        else:
            st.info("✅ **Monitor engagement**")

st.markdown("---")
st.markdown("**Built with Random Forest (ROC-AUC: 0.525) | Group 11 PPA Project**")
