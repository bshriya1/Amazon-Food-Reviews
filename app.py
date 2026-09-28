import streamlit as st
import joblib

st.set_page_config(
    page_title="Amazon Food Review Predictor",
    page_icon="🍔",
    layout="centered"
)


@st.cache_resource
def load_model():
    return joblib.load("amazon_food_review_tuned_model.joblib")

model = load_model()


st.title("🍔 Amazon Food Review Predictor")
st.write("Enter a food review and the model will predict its sentiment.")

st.divider()


review = st.text_area(
    "Enter your review:",
    placeholder="Example: The food was delicious and I really enjoyed it!",
    height=150
)


if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review first.")

    else:
        prediction = model.predict([review])[0]

        st.subheader("Prediction")

        if prediction == 2:
            st.success("😊 Positive Review")

        elif prediction == 1:
            st.warning("😐 Neutral Review")

        elif prediction == 0:
            st.error("😞 Negative Review")

        else:
            st.info(f"Predicted class: {prediction}")