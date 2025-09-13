import os
import numpy as np
import librosa
from pydub import AudioSegment

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def convert_to_wav(file_path: str) -> str:
    """
    Convert any audio file (.mp3, .m4a, .wav) into .wav format using pydub.
    Returns new wav file path.
    """
    audio = AudioSegment.from_file(file_path)
    wav_path = os.path.splitext(file_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path


def extract_features(file_path: str):
    """
    Convert file to wav (if needed) and extract MFCC features.
    """
    if not file_path.endswith(".wav"):
        file_path = convert_to_wav(file_path)

    # Load audio with librosa
    y, sr = librosa.load(file_path, sr=22050, mono=True)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    return mfcc_mean


