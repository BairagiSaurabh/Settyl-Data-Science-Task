import pickle
import streamlit as st
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')

# Load the TF-IDF vectorizer and label encoder
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Download NLTK resources
nltk.download('stopwords')
lemma = WordNetLemmatizer()

# Function to clean and preprocess the input text
def clean_text(text):
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('in')
    all_stopwords.remove('out')
    all_stopwords.remove('on')
    all_stopwords.remove('of')
    all_stopwords.remove('off')

    statement = text.lower().strip()
    statement = re.sub('[^a-zA-Z]', ' ', statement)
    statement = statement.split()
    final_statement = [lemma.lemmatize(word) for word in statement if not word in set(all_stopwords) and len(word) > 1]
    final_statement_ = ' '.join(final_statement)
    return final_statement_

# Function to make prediction
def predict_internal_status(text):
    cleaned_text = clean_text(text)
    tfidf_features = tfidf_vectorizer.transform([cleaned_text])
    pred = model.predict(tfidf_features)
    y_pred = np.argmax(pred, axis=1)
    y_pred_original = label_encoder.inverse_transform(y_pred)
    return y_pred_original[0]

# Streamlit app interface
def main():
    st.title("Internal Status Prediction")
    st.write("Enter the text to predict the internal status:")
    input_text = st.text_area("Input Text", "")
    if st.button("Predict"):
        if input_text:
            prediction_result = predict_internal_status(input_text)
            st.write("Predicted Internal Status:", prediction_result)
        else:
            st.write("Error: Input text is empty. Please provide some text.")

if __name__ == "__main__":
    main()
