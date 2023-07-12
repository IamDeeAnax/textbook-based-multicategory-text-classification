import streamlit as st
import tensorflow as tf
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.corpus import words
import wordninja
from sklearn.preprocessing import LabelEncoder

# Define a function to preprocess the data
def preprocess_text(text):
    # Remove newlines
    text = re.sub(r'\n+', ' ', text)
    # Remove hyphens and put spaces
    text = re.sub(r'-', ' ', text)
    # Remove words containing numbers
    text = re.sub(r'\b\w*\d\w*\b', ' ', text)
    # Replace one or two letter words with an empty string
    text = re.sub(r'\b\w{1,2}\b', '', text)
    # Remove Roman numerals
    text = re.sub(r'\b[IVXLCDM]+\b', ' ', text, flags=re.IGNORECASE)
    # Convert to lowercase
    text = text.lower()
    # Separate joined words
    text = ' '.join(wordninja.split(text))
    # Remove URLs
    text = re.sub(r'http\S+', ' ', text)
    # Remove any special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    # Replace duplicate word with single word
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
    # Remove punctuation
    text = re.sub(r'[^\w\s]|_', ' ', text)
    # Remove specific words
    text = re.sub(r'\b(?:one|two|use|also|would|first|fig|may|used|see|new|differennt|called|many|find|part|number|using|work|chapter|example|must|true|cos|false|within|result|much|another|figure|form|three|like|however|given)\b', " ", text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:oh|ost|coo|coa|syn|yl|lih|gre|sni|tait|al|ce|ten|elo|oid|ley|rer|se|isra|blu|lk|lu|ree|lt|lus|lu|el|line|thus|end|process|change|different|could)\b', '', text, flags=re.IGNORECASE)
    # Remove single alphabets excluding "a"
    text = re.sub(r"(?<![a-zA-Z])[^aA\s][^a-zA-Z]?(?![a-zA-Z])", "", text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load model
model_load = tf.keras.models.load_model("/Users/mac/Host_model/NLP_task 2/subject_classification_model")

# Load encoder
encoder_classes = np.load("/Users/mac/Host_model/NLP_task 2/encoder_classes.npy", allow_pickle=True)
encoder = LabelEncoder()
encoder.classes_ = encoder_classes

# Set up NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Streamlit app title
st.title("School Subject Classification")

# Description
st.markdown("## Description")
st.markdown("""
This app predicts the school subject based on the input text. 
The text is preprocessed and then classified using a trained model. 
The model can predict one of the following four categories: 
Physics, Biology, History, and Computer Science.
""")

# User input text
st.markdown("## Input Text")
input_text = st.text_area("Enter the text:", "")

# Preprocess the input text
preprocessed_text = preprocess_text(input_text)

# Button to predict subject
if st.button("Predict Subject"):
    try:
        # Show a progress bar
        with st.spinner('Predicting...'):
            # Remove stop words from the preprocessed text
            filtered_sentence = " ".join([word for word in preprocessed_text.split() if word.lower() not in stop_words])

            # Convert preprocessed text to tensor
            input_sequences = tf.constant([filtered_sentence], dtype=tf.string)

            # Use the loaded model for predictions
            prediction = model_load.predict(input_sequences)

            # Get the predicted label
            predicted_label = np.argmax(prediction, axis=-1)
            predicted_class_name = encoder.inverse_transform(predicted_label)[0]

            # Display the predicted subject
            st.success(f"The input text belongs to the category: {predicted_class_name}")
        st.success('Done!')
    # Error handling
    except Exception as e:
        st.error('Error: ' + str(e))