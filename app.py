import streamlit as st
import joblib

st.set_page_config(page_title="Food Review Predictor", page_icon="🍔")


@st.cache_resource
def load_model():
    return joblib.load("amazon_food_review_model.joblib")


model = load_model()

st.title("🍔 Food Review Predictor")
st.caption("Tell me what you think, and I'll guess the sentiment!")

review = st.text_area("💬 Your review", placeholder="The food was amazing! 😋", height=120)

if st.button("Predict Sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review 😊")
    else:
        prediction = model.predict([review])[0]

        if prediction == 2:
            st.success("😊 Positive Review!")
        elif prediction == 1:
            st.warning("😐 Neutral Review")
        elif prediction == 0:
            st.error("😞 Negative Review")
        else:
            st.info(f"Prediction: {prediction}")



