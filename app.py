import os
import uuid
import sqlite3
import streamlit as st
import numpy as np
import pyttsx3
import tempfile
import json
import librosa
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from audio_recorder_streamlit import audio_recorder

# ===============================
# CONFIGURATION
# ===============================
FFMPEG_BIN = r"C:\Users\gotte\Downloads\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_BIN
AudioSegment.converter = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_BIN, "ffprobe.exe")

DB_PATH = "database.db"
UPLOADS_DIR = "uploads"
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"

# Ensure upload directory exists
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Load Vosk speech recognition model
vosk_model = Model(VOSK_MODEL_PATH)

# Globals for FAQ search
vectorizer = TfidfVectorizer()
faq_questions = []
faq_answers = []
faq_vectors = None

# Globals for audio classification
audio_features = []
audio_labels = []

# ===============================
# Audio Feature Extraction
# ===============================
def extract_audio_features(file_path):
    """Extract features like MFCC, spectral centroid, zero crossing rate, and RMS from audio."""
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)

    feature_vector = np.hstack([
        mfcc_mean,
        spectral_centroid_mean,
        zero_crossing_rate_mean,
        rms_mean
    ])
    return feature_vector

# ===============================
# Database Helpers
# ===============================
def load_faq_from_db():
    """Load FAQ questions and answers from the database and vectorize them."""
    global faq_questions, faq_answers, faq_vectors
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT question, answer FROM faq")
        rows = c.fetchall()
        conn.close()

        if rows:
            faq_questions = [q for q, _ in rows]
            faq_answers = [a for _, a in rows]
            try:
                faq_vectors = vectorizer.fit_transform(faq_questions)
            except ValueError:
                st.warning("No FAQ questions found in the database. Please check init_db.py.")
                faq_vectors = None
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        faq_vectors = None

def load_audio_features_and_train():
    """Load labeled audio data from DB, extract features, and train KNN."""
    global audio_features, audio_labels
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT file_name, label FROM audio_data")
    rows = c.fetchall()
    conn.close()

    audio_features.clear()
    audio_labels.clear()

    # Extract features from each audio file
    for file_name, label in rows:
        file_path = os.path.join(UPLOADS_DIR, file_name)
        if os.path.exists(file_path):
            try:
                features = extract_audio_features(file_path)
                audio_features.append(features)
                audio_labels.append(label)
            except Exception as e:
                st.warning(f"Failed to extract features from {file_name}: {e}")
        else:
            st.warning(f"File missing: {file_name} (Expected at {file_path})")

    if audio_features:
        # Dynamically adjust n_neighbors
        k_value = min(3, len(audio_features))
        st.session_state['knn_model'] = KNeighborsClassifier(n_neighbors=k_value)
        st.session_state['knn_model'].fit(audio_features, audio_labels)
        return True
    else:
        st.session_state['knn_model'] = None
        return False

def predict_audio_label(file_path):
    """Predict the label of a new audio file using the trained KNN."""
    knn_model = st.session_state.get('knn_model', None)
    if knn_model is None:
        return "Model not trained"
    features = extract_audio_features(file_path)
    pred_label = knn_model.predict([features])[0]
    return pred_label

def search_answer(query):
    """Search for the most relevant FAQ answer using TF-IDF cosine similarity."""
    global faq_vectors
    if faq_vectors is None or not faq_questions:
        return "No data available."
    if not query.strip():
        return "Please provide a query."

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, faq_vectors)
    idx = np.argmax(sims)
    max_sim = sims[0][idx]

    if max_sim < 0.2:
        return "Sorry, I couldn't find a good match for your question."
    return faq_answers[idx]

# ===============================
# Audio Helpers
# ===============================
def transcribe_audio(file_path):
    """Convert speech in audio file to text using Vosk."""
    try:
        rec = KaldiRecognizer(vosk_model, 16000)
        audio = AudioSegment.from_file(file_path).set_frame_rate(16000).set_channels(1)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio.export(tmp_wav.name, format="wav")
            import wave
            wf = wave.open(tmp_wav.name, "rb")
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    results.append(res.get("text", ""))
            res = json.loads(rec.FinalResult())
            results.append(res.get("text", ""))
            wf.close()
        os.unlink(tmp_wav.name)
        return " ".join(results).strip()
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""

def text_to_speech(text, filename=None):
    """Convert text answer to speech and save as an audio file."""
    try:
        if filename is None:
            filename = os.path.join(UPLOADS_DIR, f"response_{uuid.uuid4()}.wav")

        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        engine.save_to_file(text, filename)
        engine.runAndWait()
        return filename
    except Exception as e:
        st.error(f"Error during text-to-speech conversion: {e}")
        return None

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Offline Audio Chatbot", layout="centered")
st.title("🎤 Offline Audio Chatbot")
st.markdown("---")

# Load FAQs at startup
load_faq_from_db()

# ===============================
# Train Model Section
# ===============================
if st.button("🔄 Train Model with Audio Samples"):
    with st.spinner("Extracting features and training model..."):
        success = load_audio_features_and_train()
    if success:
        st.success("Training completed successfully!")
    else:
        st.error("No audio samples found to train. Please upload labeled audio files.")

# ===============================
# Method 1: Upload Audio → Real-time Prediction
# ===============================
st.header("📂 Method 1: Upload Audio → Real-time Prediction")
uploaded_file = st.file_uploader("Upload an audio file (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"])
if uploaded_file:
    file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(file_path, format=f"audio/{uploaded_file.type.split('/')[-1]}")
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("🔄 Process Uploaded Audio"):
        knn_model = st.session_state.get('knn_model', None)
        if knn_model is None:
            st.warning("⚠ Please train the model first.")
        else:
            with st.spinner("Processing your audio..."):
                pred_label = predict_audio_label(file_path)
                query_text = transcribe_audio(file_path)
                answer = search_answer(query_text)
                audio_file = text_to_speech(answer)

            st.subheader("Response Section")
            st.write("**Predicted Label:**", pred_label)
            st.write("**Transcribed Text:**", query_text)
            st.write("**Bot Answer:**", answer)
            if audio_file:
                st.audio(audio_file, format="audio/wav")

# ===============================
# Method 2: Mic / Text → Direct Response
# ===============================
st.header("🎙 Method 2: Interaction (Mic / Text → Direct Response)")
col1, col2 = st.columns(2)

# ----------- Voice Recording Section -----------
with col1:
    st.subheader("🎤 Record Your Voice")
    audio_bytes = audio_recorder()

    if audio_bytes:
        query_path = os.path.join(UPLOADS_DIR, "recorded_query.wav")
        with open(query_path, "wb") as f:
            f.write(audio_bytes)

        st.audio(query_path, format="audio/wav")
        st.success("Voice recorded! Click 'Submit Recorded Audio' to process.")

        if st.button("🔍 Submit Recorded Audio"):
            knn_model = st.session_state.get('knn_model', None)
            if knn_model is None:
                st.warning("⚠ Please train the model first before processing voice queries.")
            else:
                with st.spinner("Processing your voice query..."):
                    # 1. Transcribe audio
                    query_text = transcribe_audio(query_path)

                    # 2. Predict label
                    pred_label = predict_audio_label(query_path)

                    # 3. Find FAQ answer
                    answer = search_answer(query_text)

                    # 4. Convert to speech
                    audio_file = text_to_speech(answer)

                st.subheader("Response Section")
                st.write("**Predicted Label:**", pred_label)
                st.write("**Transcribed Text:**", query_text)
                st.write("**Bot Answer:**", answer)
                if audio_file:
                    st.audio(audio_file, format="audio/wav")

# ----------- Text Query Section -----------
with col2:
    st.subheader("⌨ Type Your Question")
    text_query = st.text_input("Enter your question:")

    if st.button("➡️ Submit Text Query"):
        if not faq_questions:
            st.error("⚠ FAQ data not loaded. Please check your database and reload.")
        elif text_query.strip():
            with st.spinner("Processing your text query..."):
                response = search_answer(text_query)
                audio_file = text_to_speech(response)

            st.subheader("Response Section")
            st.write("**You Asked:**", text_query)
            st.write("**Bot Answer:**", response)
            if audio_file:
                st.audio(audio_file, format="audio/wav")
        else:
            st.warning("Please enter a question before submitting.")
