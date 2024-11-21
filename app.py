import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import findSimilarity
import findEmotion
import findlip_sync

# Set page config to wide layout
st.set_page_config(page_title="Dubbing LK", layout="wide")

# Load custom CSS for styling
def load_css(style):
    with open(style, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Center-align the main title and improve the look
st.markdown("<h1 style='text-align: center; color: #4CAF50; font-size: 3em;'>Dubbing LK</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #8F8F8F;'>Audio Analysis and Dubbing Tool</h3>", unsafe_allow_html=True)

# Welcome Section
st.write(""" 
#### Welcome to Dubbing LK!  
This tool allows you to **compare voice similarities**, visualize **MFCC features**, and analyze **emotions** in audio files. Let's dive into voice dubbing analysis!
""")

# Spacer for better layout
st.markdown("<br>", unsafe_allow_html=True)

# File Upload Section for similarity analysis
st.markdown("### 📁 Upload Your Audio Files:")
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h4>Base Audio File</h4>", unsafe_allow_html=True)
    uploaded_file_1 = st.file_uploader("", type=["wav", "mp3"], key="base_audio")

with col2:
    st.markdown("<h4>Tester Audio File</h4>", unsafe_allow_html=True)
    uploaded_file_2 = st.file_uploader("", type=["wav", "mp3"], key="tester_audio")

# Spacer
st.markdown("<hr>", unsafe_allow_html=True)

# Handle file uploads and temporary storage for similarity analysis
if uploaded_file_1 and uploaded_file_2:
    st.success('Files uploaded successfully!')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file_1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file_2:
        
        temp_file_1.write(uploaded_file_1.getbuffer())
        temp_file_2.write(uploaded_file_2.getbuffer())
        
        temp_file_1_path = temp_file_1.name
        temp_file_2_path = temp_file_2.name

    # Load models for both similarity and emotion detection
    try:
        similarity_model = findSimilarity.load_trained_model('/Audio_analysis/Data/voice_similarity_model.h5')
        emotion_model = findEmotion.load_trained_model('/Audio_analysis/Data/emotion_recognition_model.h5')
    except FileNotFoundError:
        st.error("Model file not found. Please ensure that the model paths are correct.")
    else:
        # Button to trigger analysis
        if st.button('🔍 Analyze'):
            st.markdown("<h4 style='text-align: center;'>Analyzing the audio files... Please wait.</h4>", unsafe_allow_html=True)

            # Emotion analysis for both audio files
            base_audio_emotion = findEmotion.predict_emotion(emotion_model, temp_file_1_path)
            tester_audio_emotion = findEmotion.predict_emotion(emotion_model, temp_file_2_path)

            # Results Section with better layout and formatting
            st.markdown("<br>", unsafe_allow_html=True)
    
            # Create two columns for emotions
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"### Base Audio Emotion: **{base_audio_emotion}**")
                findSimilarity.plot_audio_waveform(temp_file_1_path, title='Base Audio Waveform')
            with col2:
                st.write(f"### Tester Audio Emotion: **{tester_audio_emotion}**")
                findSimilarity.plot_audio_waveform(temp_file_2_path, title='Tester Audio Waveform')
           
            pitch_similarity_percentage = findSimilarity.compare_pitch(temp_file_1_path, temp_file_2_path)
            st.markdown("<br>", unsafe_allow_html=True)
            similarity_percentage = findSimilarity.check_audio_similarity(similarity_model, temp_file_1_path, temp_file_2_path)
            st.write(f"### Overall Audio similarity percentage: **{similarity_percentage:.2f}%**") 
           
            col1, col2 = st.columns(2)
            with col1:
                # Display similarity breakdown with improved visuals
                st.markdown("### 📊 Similarity Breakdown:")
                findSimilarity.plot_similarity_charts(similarity_percentage)
            with col2:
                # MFCC Variance plot section
                st.markdown("### 🎼 MFCC Variance Plot:")
                findSimilarity.plot_mfcc_variance(temp_file_1_path, temp_file_2_path)
            st.markdown("<br>", unsafe_allow_html=True)

# File Upload Section for lip-sync analysis
st.markdown("### 📁 Upload Your Audio and Video Files:")
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h4>Audio File</h4>", unsafe_allow_html=True)
    uploaded_audio_file = st.file_uploader("", type=["wav", "mp3"], key="audio_file")  # Changed key to "audio_file"

with col2:
    st.markdown("<h4>Video File</h4>", unsafe_allow_html=True)
    uploaded_video_file = st.file_uploader("", type=["mp4"], key="video_file")

# Spacer
st.markdown("<hr>", unsafe_allow_html=True)

# Handle file uploads and temporary storage for lip-sync analysis
if uploaded_audio_file and uploaded_video_file:
    st.success('Files uploaded successfully!')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        
        temp_audio_file.write(uploaded_audio_file.getbuffer())
        temp_video_file.write(uploaded_video_file.getbuffer())
        
        temp_audio_file_path = temp_audio_file.name
        temp_video_file_path = temp_video_file.name

    # Analyze Lip Sync
    if st.button('🔍 Analyze Lip Sync'):
        findlip_sync.analyze_lip_sync(temp_audio_file_path, temp_video_file_path)
            
else:
    st.warning("Please upload both audio files to proceed with the analysis.")
