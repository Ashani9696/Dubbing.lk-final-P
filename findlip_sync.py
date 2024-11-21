import librosa
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt

def segment_audio(audio_file, sr=22050):
    y, _ = librosa.load(audio_file, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    frame_times = librosa.times_like(rms, sr=sr)

    phoneme_segments = []
    threshold = np.mean(rms) * 1.2

    is_phoneme = False
    for i, value in enumerate(rms):
        if value > threshold and not is_phoneme:
            start_time = frame_times[i]
            is_phoneme = True
        elif value <= threshold and is_phoneme:
            end_time = frame_times[i]
            phoneme_segments.append((start_time, end_time))
            is_phoneme = False

    return phoneme_segments

def get_lip_movements(video_file):
    cap = cv2.VideoCapture(video_file)

    lip_movements = []
    frame_count = 0

    ret, first_frame = cap.read()
    if not ret:
        return []

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    mouth_points = np.array([[150, 200], [200, 200], [250, 200]], dtype=np.float32)  # Adjust based on your video resolution

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, mouth_points, None)

        if new_points is not None and status is not None:
            upper_lip = new_points[0]  
            lower_lip = new_points[2]
            lip_opening = np.linalg.norm(upper_lip - lower_lip)

            lip_movements.append((frame_count / cap.get(cv2.CAP_PROP_FPS), lip_opening))

        prev_gray = gray
        mouth_points = new_points  
        frame_count += 1

    cap.release()
    return lip_movements

def match_lip_sync(audio_file, video_file):
    phoneme_segments = segment_audio(audio_file)
    lip_movements = get_lip_movements(video_file)

    alignment_results = []
    phoneme_idx = 0

    for (time, opening) in lip_movements:
        if phoneme_idx >= len(phoneme_segments):
            break
        
        start, end = phoneme_segments[phoneme_idx]
        if start <= time <= end:
            mean_opening = np.mean([o for _, o in lip_movements]) if lip_movements else 0
            std_opening = np.std([o for _, o in lip_movements]) if lip_movements else 0
            dynamic_threshold = mean_opening + std_opening

            if opening > dynamic_threshold:
                alignment_results.append((time, True))
            else:
                alignment_results.append((time, False))
        
        if time > end:
            phoneme_idx += 1

    matched = sum(1 for _, match in alignment_results if match)
    sync_percentage = (matched / len(alignment_results)) * 100 if alignment_results else 0

    return sync_percentage, alignment_results, mean_opening, std_opening

def analyze_lip_sync(audio_file, video_file):
    sync_percentage, alignment_results, mean_opening, std_opening = match_lip_sync(audio_file, video_file)
    st.write(f"Lip Sync Match: {sync_percentage:.2f}%")
    st.write(f"Average Lip Opening: {mean_opening:.2f} pixels")
    st.write(f"Standard Deviation of Lip Opening: {std_opening:.2f} pixels")

    # Feedback based on sync percentage
    if sync_percentage >= 80:
        feedback = "Excellent synchronization!"
    elif sync_percentage >= 60:
        feedback = "Good synchronization, but there's room for improvement."
    else:
        feedback = "Poor synchronization, consider adjusting your audio or video."
    
    st.write(feedback)

    if alignment_results:
        times, matches = zip(*alignment_results)
        plt.figure(figsize=(12, 6))
        plt.plot(times, matches, marker='o', label="Lip Sync Match (True=Match)", color='blue')
        
        # Indicate matches and mismatches with different colors
        for i, match in enumerate(matches):
            if match:
                plt.plot(times[i], matches[i], 'go')  # Green for match
            else:
                plt.plot(times[i], matches[i], 'ro')  # Red for mismatch

        plt.axhline(0.5, color='r', linestyle='--', label='Threshold')  # Line to indicate threshold
        plt.axvline(x=np.mean(times), color='orange', linestyle='--', label='Average Time')  # Average time line
        plt.xlabel("Time (s)")
        plt.ylabel("Lip Sync Match (1=True, 0=False)")
        plt.title("Lip Sync Match Over Time")
        plt.ylim(-0.1, 1.1)
        plt.legend()
        st.pyplot(plt)
    else:
        st.write("No lip movements detected.")