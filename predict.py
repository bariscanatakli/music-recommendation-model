import os
import argparse
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from models.advanced_models import create_improved_model
from utils import setup_gpu_memory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict music genre from audio files')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--audio_file', type=str, required=True, help='Path to audio file')
    parser.add_argument('--top_k', type=int, default=3, help='Show top K predictions')
    return parser.parse_args()


def load_audio(audio_path, duration=30, sr=22050):
    """Load and preprocess audio file."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        
        # Resize to 128x128 (matching training dimensions)
        if mel_db.shape[1] > 128:
            mel_db = mel_db[:, :128]
        else:
            pad_width = 128 - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), 'constant')
        
        # Add batch and channel dimensions
        mel_db = mel_db.reshape(1, mel_db.shape[0], mel_db.shape[1], 1)
        
        return mel_db
    
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None


def plot_spectrogram(audio_path):
    """Plot the spectrogram of the audio file."""
    y, sr = librosa.load(audio_path, sr=22050, duration=30)
    
    plt.figure(figsize=(10, 4))
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup GPU memory growth
    setup_gpu_memory()
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    try:
        model = tf.keras.models.load_model(args.model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Define genre names (must match training order)
    genre_names = ['blues', 'classical', 'country', 'disco', 'hip-hop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Process audio file
    print(f"Processing audio file: {args.audio_file}")
    audio_features = load_audio(args.audio_file)
    
    if audio_features is None:
        return
    
    # Make prediction
    print("Making prediction...")
    predictions = model.predict(audio_features)[0]
    
    # Get top K predictions
    top_indices = np.argsort(predictions)[-args.top_k:][::-1]
    top_genres = [genre_names[i] for i in top_indices]
    top_probs = [predictions[i] * 100 for i in top_indices]
    
    # Display results
    print("\n===== Genre Prediction Results =====")
    print(f"File: {os.path.basename(args.audio_file)}")
    print("\nTop predictions:")
    for i in range(args.top_k):
        print(f"{i+1}. {top_genres[i]}: {top_probs[i]:.2f}%")
    
    # Plot spectrogram (optional)
    plot_choice = input("\nWould you like to see the spectrogram? (y/n): ")
    if plot_choice.lower() == 'y':
        plot_spectrogram(args.audio_file)


if __name__ == "__main__":
    main()