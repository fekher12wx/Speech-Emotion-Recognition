"""
Emotion Prediction Module
Contains functions for loading model and making predictions
"""
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

def extract_features(file_path, duration=3, offset=0.5):
    """Enhanced feature extraction - same as training"""
    try:
        audio, sr = librosa.load(file_path, duration=duration, offset=offset, sr=22050)

        # MFCC + Deltas
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta_mean = np.mean(mfccs_delta.T, axis=0)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)

        # Mel
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_mean = np.mean(mel.T, axis=0)
        mel_std = np.std(mel.T, axis=0)

        # Contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)

        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)

        # Prosodic features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        rms = np.mean(librosa.feature.rms(y=audio))

        return np.concatenate([
            mfccs_mean, mfccs_std, mfccs_delta_mean,
            chroma_mean, chroma_std,
            mel_mean, mel_std,
            contrast_mean, tonnetz_mean,
            [spectral_centroid, spectral_rolloff, zcr, rms]
        ])
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

class EmotionPredictor:
    """Class to handle emotion prediction"""
    
    def __init__(self, model_path=None, scaler_path=None, encoder_path=None):
        """Initialize the predictor with model and preprocessing objects"""
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
        # Try to load model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Try default paths
            for path in ['best_emotion_model_v2.keras', 'best_emotion_model.keras']:
                if os.path.exists(path):
                    self.load_model(path)
                    break
        
        # Try to load scaler and encoder
        if scaler_path and encoder_path:
            self.load_preprocessing(scaler_path, encoder_path)
        else:
            self.load_preprocessing('scaler.pkl', 'label_encoder.pkl')
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
    
    def load_preprocessing(self, scaler_path, encoder_path):
        """Load scaler and label encoder"""
        try:
            if os.path.exists(scaler_path) and os.path.exists(encoder_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print(f"✅ Preprocessing objects loaded")
            else:
                # Create default ones
                self.scaler = StandardScaler()
                emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(emotions)
                print("⚠️ Using default preprocessing (model may not work correctly without trained scaler/encoder)")
        except Exception as e:
            print(f"❌ Error loading preprocessing: {str(e)}")
    
    def predict(self, audio_file_path):
        """Predict emotion from audio file path"""
        if self.model is None:
            return None, None, None
        
        # Extract features
        features = extract_features(audio_file_path)
        if features is None:
            return None, None, None
        
        # Preprocess
        features_scaled = self.scaler.transform([features])
        features_reshaped = np.expand_dims(features_scaled, axis=2)
        
        # Predict
        prediction = self.model.predict(features_reshaped, verbose=0)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_emotions = self.label_encoder.classes_[top_3_indices]
        top_3_confidences = prediction[0][top_3_indices] * 100
        
        # Main prediction
        emotion = top_3_emotions[0]
        confidence = top_3_confidences[0]
        
        return emotion, confidence, list(zip(top_3_emotions, top_3_confidences))

def save_preprocessing_objects(scaler, label_encoder, scaler_path='scaler.pkl', encoder_path='label_encoder.pkl'):
    """Save scaler and label encoder for later use"""
    try:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"✅ Preprocessing objects saved to {scaler_path} and {encoder_path}")
    except Exception as e:
        print(f"❌ Error saving preprocessing objects: {str(e)}")

if __name__ == "__main__":
    # Example usage
    predictor = EmotionPredictor()
    if predictor.model:
        print("Model loaded successfully!")
        print("Use predictor.predict('audio_file.wav') to make predictions")

