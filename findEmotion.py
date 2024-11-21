import numpy as np
import librosa
from tensorflow.keras.models import load_model # type: ignore

# Load the trained model
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# Function to extract MFCC features
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Preprocessing function
def preprocess_audio(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
    mfcc = np.expand_dims(mfcc, axis=2)  # Add channel dimension for LSTM
    return mfcc

# Function to predict emotion for a given audio file
def predict_emotion(model, audio_file):
    processed_audio = preprocess_audio(audio_file)
    prediction = model.predict(processed_audio)

    # Define emotion labels
    emotion_labels = ['Angry', 'Sad', 'ps', 'Surprise', 'Neutral', 'Fear', 'Happy']

    # Get the predicted emotion
    predicted_emotion = emotion_labels[np.argmax(prediction)]
    return predicted_emotion
