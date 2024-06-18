import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.corpus import words
import wordninja
from sklearn.preprocessing import LabelEncoder
import base64
import os

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

# Define the number of output classes 
num_classes = 4

# Define your model architecture using the pre-trained embedding
embedding_layer = hub.KerasLayer(
    "https://tfhub.dev/google/nnlm-en-dim50/2",
    input_shape=[],
    dtype=tf.string,
    trainable=True,
)

model = tf.keras.Sequential([
    embedding_layer,
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Define the paths for your files
weights_path = 'Streamlit/model/subject_classification_model_weights.h5'
encoder_path = 'Streamlit/model/encoder_classes.npy'
background_image_path = 'Streamlit/background.png'


# Load the model weights from the local path
try:
    model.load_weights(weights_path)
    encoder_classes = np.load(encoder_path, allow_pickle=True)
    encoder = LabelEncoder()
    encoder.classes_ = encoder_classes
    
except FileNotFoundError as e:
    error_message = f"File not found: {str(e)}"
    st.error(error_message)

except AttributeError as e:
    error_message = f"Attribute error: {str(e)}"
    st.error(error_message)

except Exception as e:
    error_message = f"Error loading the encoder: {str(e)}"
    st.error(error_message)

# Set up NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local(background_image_path)

# Custom CSS to change text color, size, and alignment
st.markdown(
    """
    <style>
    .title {
        color: black;
        font-size: 20px;
        text-align: left;
        font-weight: bold;
    }
    .text {
        color: black;
        font-size: 20px;
        text-align: center;
    }
    .heading {
        color: black;
        font-size: 30px;
        text-align: center;
        font-weight: bold;
    }
    .stTextArea label {
        color: black !important;
        font-size: 30px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Streamlit app title
st.markdown("<h1 class='heading'>SCHOOL SUBJECT CLASSIFICATION</h1>", unsafe_allow_html=True)

# Description
st.markdown("<h2 class='title'>Description</h2>", unsafe_allow_html=True)
st.markdown(
    """
    <p class='text'>
    This app predicts the school subject based on the input text. 
    The text is preprocessed and then classified using a trained model. 
    The model can predict one of the following four categories: 
    Physics, Biology, History, and Computer Science.
    </p>
    """, 
    unsafe_allow_html=True
)

# User input text
st.markdown("<h2 class='title'>Input Text</h2>", unsafe_allow_html=True)
input_text = st.text_area("Enter the text:", "")

# Button to predict subject
if st.button("Predict Subject"):
    # Check if the input text is empty
    if input_text.strip() == "":
        st.warning('Please enter a text.')
    else:
        try:
            # Show a progress bar
            with st.spinner('Predicting...'):
                # Preprocess the input text
                preprocessed_text = preprocess_text(input_text)
                
                # Remove stop words from the preprocessed text
                filtered_sentence = " ".join([word for word in preprocessed_text.split() if word.lower() not in stop_words])

                # Convert preprocessed text to tensor
                input_sequences = tf.constant([filtered_sentence], dtype=tf.string)

                # Use the loaded model for predictions
                prediction = model.predict(input_sequences)

                # Get the predicted label
                predicted_label = np.argmax(prediction, axis=-1)
                predicted_class_name = encoder.inverse_transform(predicted_label)[0]

                # Get the confidence of the prediction
                confidence = np.max(prediction)

                # Set a confidence threshold
                confidence_threshold = 0.5

                # Check if the confidence level is below the threshold
                if confidence < confidence_threshold:
                    st.warning('The input text does not belong to any category.')
                else:
                    # Display the predicted subject with confidence
                    st.success(f"The input text belongs to the category: {predicted_class_name}")
                    # st.info(f"Confidence level: {confidence:.2f}")
            st.success('Done!')
        # Error handling
        except Exception as e:
            st.error('Error: ' + str(e))
