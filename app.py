import streamlit as st
import numpy as np
import librosa
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import tempfile
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .emotion-result-card {
        padding: 2.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .emotion-result-card h2 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .emotion-result-card h3 {
        font-size: 1.5rem;
        font-weight: 400;
        opacity: 0.95;
    }
    
    .upload-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 3rem;
        border-radius: 20px;
        border: 3px dashed #667eea;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #764ba2;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
    }
    
    .emotion-card {
        padding: 1.5rem;
        border-radius: 15px;
        background: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 5px solid #667eea;
    }
    
    .emotion-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        padding: 1.5rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .sidebar-content {
        padding: 1rem 0;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
    }
    
    .emotion-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        margin: 0.5rem;
        font-size: 1.2rem;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    .success-banner {
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    possible_paths = [
        'best_model.keras',
        'best_emotion_model.keras',
        'best_emotion_model_v2.keras',
        'best_model_v2.keras'
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        return None, "No model file found"
    
    try:
        model = keras.models.load_model(model_path, compile=False)
        try:
            model.compile(
                loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy']
            )
        except:
            pass
        return model, None
    except Exception as e:
        error_msg = str(e)
        try:
            model = keras.models.load_model(model_path, compile=False, safe_mode=False)
            return model, None
        except Exception as e2:
            return None, f"Error loading model: {error_msg}\n\nAlso tried safe_mode=False: {str(e2)}"

@st.cache_resource
def load_preprocessing():
    """Load scaler and label encoder"""
    scaler_path = 'scaler.pkl'
    encoder_path = 'label_encoder.pkl'
    
    scaler = None
    label_encoder = None
    
    if os.path.exists(scaler_path) and os.path.exists(encoder_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
        except Exception as e:
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f, encoding='latin1')
                with open(encoder_path, 'rb') as f:
                    label_encoder = pickle.load(f, encoding='latin1')
            except Exception as e2:
                st.warning(f"⚠️ Could not load preprocessing files: {str(e)}. Using defaults (may affect accuracy).")
    
    if scaler is None:
        scaler = StandardScaler()
    
    if label_encoder is None:
        emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        label_encoder = LabelEncoder()
        label_encoder.fit(emotions)
    
    return scaler, label_encoder

def extract_features(file_path, duration=3, offset=0.5):
    """Extract audio features for emotion prediction (193 features total)"""
    try:
        audio, sr = librosa.load(file_path, duration=duration, offset=offset, sr=22050)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr).T, axis=0)
        return np.concatenate([mfccs, chroma, mel, contrast, tonnetz])
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def predict_emotion(audio_file, model, scaler, label_encoder):
    """Predict emotion from audio file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            if isinstance(audio_file, BytesIO):
                tmp_file.write(audio_file.read())
            else:
                tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
        
        features = extract_features(tmp_path)
        if features is None:
            return None, None, None
        
        if not hasattr(scaler, 'mean_') or scaler.mean_ is None:
            st.warning("⚠️ Scaler not fitted. Using raw features (may affect accuracy).")
            features_scaled = [features]
        else:
            features_scaled = scaler.transform([features])
        
        features_reshaped = np.expand_dims(features_scaled, axis=2)
        prediction = model.predict(features_reshaped, verbose=0)
        
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_emotions = label_encoder.classes_[top_3_indices]
        top_3_confidences = prediction[0][top_3_indices] * 100
        
        emotion = top_3_emotions[0]
        confidence = top_3_confidences[0]
        
        os.unlink(tmp_path)
        return emotion, confidence, list(zip(top_3_emotions, top_3_confidences))
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

EMOTION_EMOJIS = {
    'angry': '😠',
    'calm': '😌',
    'disgust': '🤢',
    'fearful': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprised': '😲'
}

def main():
    # Hero Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">🎤 Speech Emotion Recognition</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Discover emotions hidden in speech using advanced AI</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a **deep learning model** to recognize emotions in speech audio.
        
        ### Supported Emotions
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin: 1rem 0;">
            <div style="padding: 0.5rem; background: #fff3cd; border-radius: 8px; text-align: center;">
                <span style="font-size: 1.5rem;">😠</span><br>Angry
            </div>
            <div style="padding: 0.5rem; background: #d1ecf1; border-radius: 8px; text-align: center;">
                <span style="font-size: 1.5rem;">😌</span><br>Calm
            </div>
            <div style="padding: 0.5rem; background: #f8d7da; border-radius: 8px; text-align: center;">
                <span style="font-size: 1.5rem;">🤢</span><br>Disgust
            </div>
            <div style="padding: 0.5rem; background: #d4edda; border-radius: 8px; text-align: center;">
                <span style="font-size: 1.5rem;">😨</span><br>Fearful
            </div>
            <div style="padding: 0.5rem; background: #fff3cd; border-radius: 8px; text-align: center;">
                <span style="font-size: 1.5rem;">😊</span><br>Happy
            </div>
            <div style="padding: 0.5rem; background: #e2e3e5; border-radius: 8px; text-align: center;">
                <span style="font-size: 1.5rem;">😐</span><br>Neutral
            </div>
            <div style="padding: 0.5rem; background: #cfe2ff; border-radius: 8px; text-align: center;">
                <span style="font-size: 1.5rem;">😢</span><br>Sad
            </div>
            <div style="padding: 0.5rem; background: #fff3cd; border-radius: 8px; text-align: center;">
                <span style="font-size: 1.5rem;">😲</span><br>Surprised
            </div>
        </div>
        
        ### 📋 Instructions
        
        1. **Upload** an audio file (WAV, MP3, M4A, or FLAC)
        2. **Listen** to the audio (optional)
        3. **Click** "Predict Emotion"
        4. **View** detailed results with confidence scores
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.header("⚙️ Settings")
        show_audio = st.checkbox("🔊 Show audio player", value=True)
        show_details = st.checkbox("📊 Show detailed predictions", value=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.spinner("Loading model..."):
        model_result = load_model()
        scaler, label_encoder = load_preprocessing()
    
    if isinstance(model_result, tuple):
        model, error_msg = model_result
    else:
        model = model_result
        error_msg = None
    
    if model is None:
        model_files = ['best_emotion_model_v2.keras', 'best_emotion_model.keras', 'best_model.keras', 'best_model_v2.keras']
        found_files = [f for f in model_files if os.path.exists(f)]
        
        if found_files:
            st.error(f"""
            ## ❌ Model Loading Failed
            
            Model file found: **{found_files[0]}** but failed to load.
            
            **Error Details:**
            ```
            {error_msg if error_msg else "Unknown error"}
            ```
            
            **Possible Solutions:**
            1. **Keras Version Mismatch**: The model was saved with a different Keras version
               - Try: `pip install --upgrade tensorflow keras`
            2. **Model File Corrupted**: The file might be incomplete
               - Re-download from Colab
            3. **Check file size**: Model should be ~25-100 MB
               - Current file: {os.path.getsize(found_files[0]) / (1024*1024):.2f} MB
            
            **Quick Fix:**
            - Re-download `best_model.keras` from Colab
            - Make sure the file download completed fully
            """)
        else:
            st.error("""
            ## ❌ Model Not Found
            
            The trained model file is missing. Please ensure you have one of these files in your project directory:
            - `best_emotion_model_v2.keras`
            - `best_emotion_model.keras`
            - `best_model.keras` (Colab default)
            
            **To fix this:**
            1. Train your model using `deep_project.py`
            2. Or download your trained model file to this directory
            
            The app will not function without a trained model.
            """)
        st.info("💡 **Note:** Once you have the model file, refresh this page to use the app.")
        return
    
    # Success Banner
    st.markdown('<div class="success-banner">✅ Model loaded successfully! Ready to analyze emotions.</div>', unsafe_allow_html=True)
    
    # Upload Section
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 📁 Upload Audio File")
        uploaded_file = st.file_uploader(
            "",
            type=['wav', 'mp3', 'm4a', 'flac'],
            help="Upload an audio file to analyze emotions",
            label_visibility="collapsed"
        )
    
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if show_audio:
                st.markdown("<br>", unsafe_allow_html=True)
                st.audio(uploaded_file, format='audio/wav')
                st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🔮 Predict Emotion", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing audio features..."):
                    emotion, confidence, top_3 = predict_emotion(uploaded_file, model, scaler, label_encoder)
                
                if emotion is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("---")
                    
                    # Main Result Card
                    emoji = EMOTION_EMOJIS.get(emotion.lower(), '🎭')
                    st.markdown(f"""
                    <div class="emotion-result-card">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">{emoji}</div>
                        <h2>{emotion.upper()}</h2>
                        <h3>Confidence: {confidence:.2f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Stats Container
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.markdown(f"""
                        <div class="stat-item">
                            <div class="stat-value">{emoji}</div>
                            <div class="stat-label">Emotion</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_stat2:
                        st.markdown(f"""
                        <div class="stat-item">
                            <div class="stat-value">{confidence:.1f}%</div>
                            <div class="stat-label">Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_stat3:
                        confidence_level = "High" if confidence >= 70 else "Moderate" if confidence >= 50 else "Low"
                        st.markdown(f"""
                        <div class="stat-item">
                            <div class="stat-value">{confidence_level}</div>
                            <div class="stat-label">Level</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence Indicator
                    if confidence >= 70:
                        st.success(f"✅ High confidence prediction ({confidence:.2f}%)")
                    elif confidence >= 50:
                        st.warning(f"⚠️ Moderate confidence prediction ({confidence:.2f}%)")
                    else:
                        st.info(f"ℹ️ Low confidence prediction ({confidence:.2f}%)")
                    
                    # Detailed Predictions
                    if show_details and top_3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("### 📊 Top 3 Predictions")
                        for i, (em, conf) in enumerate(top_3, 1):
                            emoji_item = EMOTION_EMOJIS.get(em.lower(), '🎭')
                            col_pred1, col_pred2 = st.columns([3, 1])
                            with col_pred1:
                                st.markdown(f"<div style='padding: 0.5rem 0; font-size: 1.1rem; font-weight: 600;'>{i}. {emoji_item} {em.capitalize()}</div>", unsafe_allow_html=True)
                            with col_pred2:
                                st.markdown(f"<div style='text-align: right; padding: 0.5rem 0; font-size: 1.1rem; color: #667eea; font-weight: 700;'>{conf:.2f}%</div>", unsafe_allow_html=True)
                            st.progress(conf / 100, text="")
                            st.markdown("<br>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p style='color: #6c757d;'>Speech Emotion Recognition using Deep Learning | Fakher ben yahya</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

