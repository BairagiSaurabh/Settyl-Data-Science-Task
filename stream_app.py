import streamlit as st
import requests

# Streamlit UI layout
st.title("Internal Status Prediction")
st.write("Enter the text to predict internal status:")

# Input text box
input_text = st.text_area("Input Text", "")

# Function to send request to FastAPI server and get prediction
def get_prediction(input_text):
    url = "http://127.0.0.1:8000/predict"
    data = {"text": input_text}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            prediction = response.json()["predicted_internal_status"]
            return prediction
        else:
            st.error("Error occurred while processing the request. Please try again.")
            return None
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        return None

# Predict button
if st.button("Predict"):
    if input_text.strip() == "":
        st.error("Please enter some text.")
    else:
        prediction = get_prediction(input_text)
        if prediction:
            st.success(f"Predicted internal status: {prediction}")
