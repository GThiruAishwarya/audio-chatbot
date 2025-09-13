import pyttsx3
import os

os.makedirs("uploads", exist_ok=True)

engine = pyttsx3.init()
samples = {
    "hello.wav": "Hello, this is a test voice.",
    "goodbye.wav": "Goodbye, see you later.",
    "thanks.wav": "Thank you for using the audio chatbot."
}

for filename, text in samples.items():
    filepath = os.path.join("uploads", filename)
    engine.save_to_file(text, filepath)

engine.runAndWait()
print("Sample audio files generated in uploads/ folder.")
