from flask import Flask, render_template, request, redirect, url_for
import os
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
import numpy as np
import torch

app = Flask(__name__)
app.static_folder = 'static'

# Configure upload folder (adjust as needed)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load speaker recognition model (replace with your model path)
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)


def preprocess_audio(file_path):
    """
    Loads and preprocesses audio signal, ensuring compatibility with the model.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        torch.Tensor: Preprocessed audio signal.
    """

    signal, fs = torchaudio.load(file_path)

    # Normalize audio (assuming model expects 16kHz sampling rate)
    if fs != 16000:
        signal = torchaudio.transforms.Resample(fs, 16000)(signal)

    # Ensure single channel
    if signal.shape[1] > 1:
        signal = signal[:, 0]

    # Reshape to ensure valid padding (adjust based on your model requirements)
    signal = signal.view(1, -1, 1)  # Reshape to [1, num_frames, 1]

    return signal.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


@app.route('/')
def index():
    """Renders the main HTML template for user interaction."""
    return render_template('index.html')


@app.route('/save_audio', methods=['POST'])
def save_audio():
    """Handles audio file uploads, performs speaker verification, and renders results."""
    try:
        # Fetch uploaded files
        file1 = request.files['file1']
        file2 = request.files['file2']

        # Check file extensions (example)
        if not (os.path.splitext(file1.filename)[1].lower() in ('.wav', '.mp3')):
            return render_template('error.html', error="Invalid audio format for file 1. Only .wav and .mp3 formats are allowed.")
        if not (os.path.splitext(file2.filename)[1].lower() in ('.wav', '.mp3')):
            return render_template('error.html', error="Invalid audio format for file 2. Only .wav and .mp3 formats are allowed.")

        # Save uploaded audio files
        file1_path = os.path.join(app.config['UPLOAD_FOLDER'], 'file1.wav')
        file2_path = os.path.join(app.config['UPLOAD_FOLDER'], 'file2.wav')

        file1.save(file1_path)
        file2.save(file2_path)

        # Preprocess audio signals
        wav1 = preprocess_audio(file1_path)
        wav2 = preprocess_audio(file2_path)

        # **Solution 1: Utilize Model Outputs (if applicable):**
        try:
            # Assuming the 'verify' method returns speaker embeddings
            embeddings1, embeddings2, _ = verification.verify(wav1=wav1, wav2=wav2)

            # Compare embeddings using cosine similarity (example)
            similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)

        except AttributeError:  # Handle case where 'embedding_layer' is not available
            # **Solution 2: Alternative Approach (if no embedding layer):**
            print("Warning: 'embedding_layer' attribute not found. Using alternative approach (if implemented).")
           
            similarity = 0.5

        threshold = 0.505  # Adjust threshold as needed

        if similarity >= threshold:
            result = "The speakers are likely the same."
        else:
            result = "The speakers are likely different."

        return render_template('index.html', result=result)

    except Exception as e:
        # Handle any potential errors that occur
        return render_template('error.html', error=f"An error occurred: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False for production use
