import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "new_next_word_lstm.h5"
TOKENIZER_PATH = "new_tokenizer.pickle"


@st.cache_resource
def load_prediction_model():
    return load_model(MODEL_PATH)


@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


model = load_prediction_model()
tokenizer = load_tokenizer()


def predict_next_word(model, tokenizer, text, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([text])[0]
    if not sequence:
        return None

    sequence = pad_sequences([sequence], maxlen=max_sequence_len, padding="pre", truncating="pre")
    predicted = model.predict(sequence, verbose=0)
    predicted_word_index = np.argmax(predicted)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title("Next Word Prediction")
text_input = st.text_input("Enter a sequence of words:")
if st.button("Predict"):
    if text_input:
        max_sequence_len = model.input_shape[1]
        predicted_word = predict_next_word(model, tokenizer, text_input, max_sequence_len)
        if predicted_word:
            st.write(f"Predicted next word: {predicted_word}")
        else:
            st.write("Could not predict the next word.")
    else:
        st.write("Please enter a sequence of words.")

st.write("This app uses a pre-trained model to predict the next word in a given sequence of words. The model was trained on a large corpus of text data and can provide suggestions based on the input provided by the user.")

st.write("developed by Aashutosh Kumar Bhardwaj")
