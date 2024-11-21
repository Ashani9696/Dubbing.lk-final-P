# Import required libraries
import numpy as np
import librosa
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import pearsonr
# 1. Load the saved model
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# 2. Extract MFCC features
def extract_mfcc(audio_file, sr=22050, n_mfcc=128, max_pad_len=400):
    y, _ = librosa.load(audio_file, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate the MFCCs to ensure consistent shape
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]

    # Transpose to have shape (max_pad_len, n_mfcc)
    mfccs = mfccs.T
    return mfccs

# 3. Check similarity between two audio files
def check_audio_similarity(model, audio_file_1, audio_file_2, max_pad_len=400):
    # Extract MFCC features
    mfcc_a = extract_mfcc(audio_file_1, max_pad_len=max_pad_len)
    mfcc_b = extract_mfcc(audio_file_2, max_pad_len=max_pad_len)

    # Ensure that the shapes are consistent
    mfcc_a_expanded = np.expand_dims(mfcc_a, axis=0)
    mfcc_b_expanded = np.expand_dims(mfcc_b, axis=0)

    # Convert to tensors to avoid retracing
    mfcc_a_tensor = tf.convert_to_tensor(mfcc_a_expanded, dtype=tf.float32)
    mfcc_b_tensor = tf.convert_to_tensor(mfcc_b_expanded, dtype=tf.float32)

    # Predict similarity
    prediction = model.predict([mfcc_a_tensor, mfcc_b_tensor])

    # Output similarity percentage
    similarity_score = prediction[0][0]
    similarity_percentage = similarity_score * 100
    return similarity_percentage


def extract_pitch(audio_file, sr=22050):
    y, _ = librosa.load(audio_file, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Get the highest pitch for each frame
    pitch_values = [np.max(pitches[:, t]) for t in range(pitches.shape[1])]
    
    # Filter out zero values
    pitch_values = [pitch for pitch in pitch_values if pitch > 0]
    return pitch_values
# 4. Plot basic visualizations (Bar chart & Pie chart)

def plot_similarity_charts(similarity_percentage):
    # Bar chart for similarity
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted size

    # Pie chart for similarity vs difference
    ax = plt.subplot(1, 2, 2)
    labels = ['Similar', 'Different']
    sizes = [similarity_percentage, 100 - similarity_percentage]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)  # explode first slice

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Audio Similarity Breakdown')

    plt.tight_layout()

    # Display plot in Streamlit
    st.pyplot(fig)

def plot_mfcc_variance(audio_file_1, audio_file_2, max_pad_len=400):
    # Extract MFCCs for both files
    mfcc_a = extract_mfcc(audio_file_1, max_pad_len=max_pad_len)
    mfcc_b = extract_mfcc(audio_file_2, max_pad_len=max_pad_len)

    # Calculate the variance (difference) between MFCCs
    mfcc_diff = np.abs(mfcc_a - mfcc_b)

    # Create figure for plotting
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjusted size

    # Use 'coolwarm' for a basic color map: blue means low variance, red means high variance
    cax = ax.imshow(mfcc_diff.T, aspect='auto', origin='lower', cmap='coolwarm')

    # Add title and labels
    ax.set_title('Difference in MFCC Features Between Audio Files', fontsize=16)
    ax.set_xlabel('Time (frames)', fontsize=12)
    ax.set_ylabel('MFCC Coefficients (frequency bands)', fontsize=12)

    # Add grid lines for better readability
    ax.grid(True, color='white', linestyle='--', linewidth=0.5)

    # Add color bar to indicate the magnitude of the differences
    cbar = fig.colorbar(cax)
    cbar.set_label('Magnitude of Variance', fontsize=12)

    # Label color meaning on the color bar
    cbar_ticks = [mfcc_diff.min(), (mfcc_diff.min() + mfcc_diff.max()) / 2, mfcc_diff.max()]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(['Low Difference (Blue)', 'Medium', 'High Difference (Red)'])

    # Labeling ticks on the axes
    ax.set_xticks(np.linspace(0, max_pad_len, num=10))
    ax.set_xticklabels([f'{int(i)}' for i in np.linspace(0, max_pad_len, num=10)])
    ax.set_yticks(np.linspace(0, mfcc_diff.shape[1] - 1, num=10))
    ax.set_yticklabels([f'{int(i)}' for i in np.linspace(1, mfcc_diff.shape[1], num=10)])

    plt.tight_layout()

    # Display plot in Streamlit
    st.pyplot(fig)


def plot_audio_waveform(audio_file, title):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=22050)
    
    plt.figure(figsize=(6, 4))
    plt.plot(np.linspace(0, len(y) / sr, num=len(y)), y)
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.xlim(0, len(y) / sr)  # Set x limits to audio duration
    plt.grid(True)

    # Display plot in Streamlit
    st.pyplot(plt)  # Render the plot using Streamlit
    plt.close()  # Close the plot to prevent it from being displayed again

from collections import Counter

def categorize_pitch(pitch_values):
    # Define thresholds for Low, Medium, and High pitch levels
    low_threshold = 100    # Lower range of human pitch frequencies (in Hz)
    high_threshold = 300   # Upper range for many voices (in Hz)

    # Categorize each pitch level
    categories = []
    for pitch in pitch_values:
        if pitch < low_threshold:
            categories.append("Low")
        elif low_threshold <= pitch < high_threshold:
            categories.append("Medium")
        else:
            categories.append("High")
    
    return categories

def overall_pitch_category(categories):
    # Calculate the most frequent pitch category
    category_counts = Counter(categories)
    predominant_category = category_counts.most_common(1)[0][0]
    return predominant_category

def compare_pitch(audio_file_1, audio_file_2):
    # Extract pitch values for both audio files
    pitch_a = extract_pitch(audio_file_1)
    pitch_b = extract_pitch(audio_file_2)

    # Categorize pitch levels
    pitch_a_categories = categorize_pitch(pitch_a)
    pitch_b_categories = categorize_pitch(pitch_b)

    # Determine the overall pitch level for each file
    overall_pitch_a = overall_pitch_category(pitch_a_categories)
    overall_pitch_b = overall_pitch_category(pitch_b_categories)

    # Display overall pitch levels
    st.write("### Overall Pitch Level for Base Audio:", overall_pitch_a)
    st.write("### Overall Pitch Level for Tester Audio:", overall_pitch_b)

    # Optional: Plot categorized pitch levels over time
    plt.figure(figsize=(12, 6))
    plt.plot(pitch_a, label='Base Audio', color='blue')
    plt.plot(pitch_b, label='Tester Audio', color='orange')
    plt.xlabel('Time (frames)')
    plt.ylabel('Pitch (Hz)')
    plt.legend()
    plt.title('Pitch Levels Over Time')
    plt.grid(True)
    
    # Display the plot in Streamlit
    st.pyplot(plt)
    plt.close()

    # Return overall pitch levels if needed for further use
    return overall_pitch_a, overall_pitch_b




    # # Plot the pitch comparison
    # plt.figure(figsize=(12, 6))
    # plt.plot(pitch_a_padded, color='blue', label='Pitch of Audio File 1')
    # plt.plot(pitch_b_padded, color='orange', label='Pitch of Audio File 2')
    # plt.title(f'Pitch Comparison between Two Audio Files\nSimilarity: {similarity_percentage:.2f}%', fontsize=14)
    # plt.xlabel('Time (frames)')
    # plt.ylabel('Pitch (Hz)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # st.pyplot(plt)  # Render the plot using Streamlit
    # plt.close()  

    # return similarity_percentage

# Example usage remains the same

# Example usage:
# model_path = '/Data/voice_similarity_model.h5'  # Path to your saved model
# load_trained_model(model_path)

# audio_file_1 = '/content/drive/MyDrive/content/LJ004-0110.wav'  # Replace with your actual file path
# audio_file_2 = '/content/drive/MyDrive/content/670ba912a546d.wav'  # Replace with your actual file path

# # Check similarity
# similarity = check_audio_similarity(model, audio_file_1, audio_file_2)
# print(f"Similarity between the two audio files: {similarity:.2f}%")

# Plot basic visualizations
# plot_similarity_charts(similarity)
# plot_mfcc_variance(audio_file_1, audio_file_2)
