import pyttsx3
import tempfile


def generate_response_text(model, mfcc):
    """
    Predict the closest answer given an MFCC input.
    """
    if not model:
        return "⚠️ Model not trained yet."
    return model.predict([mfcc])[0]


def text_to_speech(text):
    """
    Convert text to speech (using pyttsx3) and return temp wav file path.
    """
    engine = pyttsx3.init()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    engine.save_to_file(text, temp_file.name)
    engine.runAndWait()
    return temp_file.name
