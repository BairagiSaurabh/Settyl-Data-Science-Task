import pickle
from fastapi import FastAPI
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
#from tensorflow.keras.models import load_model
from keras.models import load_model
import uvicorn


app = FastAPI()

nltk.download('stopwords')
lemma = WordNetLemmatizer()

class TextData(BaseModel):
    text: str

def clean_text(text):

    all_stopwords = stopwords.words('english')  # this consists all the stopwords, which will be removed later.

    # Removing the following words from list containing stopwords
    all_stopwords.remove('in')
    all_stopwords.remove('out')
    all_stopwords.remove('on')
    all_stopwords.remove('of')
    all_stopwords.remove('off')

    statement = text.lower().strip()
    statement = re.sub('[^a-zA-Z]', ' ', statement)  # replacing whatever isn't letters by an empty string
    statement = statement.split()  # forming list of words in a given review
    final_statement = [lemma.lemmatize(word) for word in statement if
                        not word in set(all_stopwords) and len(word) > 1]
    final_statement_ = ' '.join(final_statement)  # joining the words and forming the review again without stopwords
    return final_statement_


model = load_model('model.h5')

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

default_message = '''
Welcome to internalStatus prediction model. Here are some sample test cases you can use for prediction (Gate in, Unloaded PRSJU, Gate out, Empty, Load on MSC ORION / 227N, Empty to Shipper,Import Unloaded from Rail)'''.strip()

@app.get('/')
async def index():
    return {'message': default_message}


# Define your API endpoints and functions here
@app.post("/predict")
async def predict_internal_status(text_data: TextData):
    cleaned_text = clean_text(text_data.text) # Preprocess the input text

    tfidf_features = tfidf_vectorizer.transform([cleaned_text])  # Transform the preprocessed text using TF-IDF vectorizer
    pred = model.predict(tfidf_features)  # Predict the internal status using the model
    y_pred = np.argmax(pred, axis=1) # Convert predicted labels to original class labels
    y_pred_original = label_encoder.inverse_transform(y_pred)

    return {"predicted_internal_status": y_pred_original[0]}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn app:app --reload
