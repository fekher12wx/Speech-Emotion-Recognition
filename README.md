# 🎤 Speech Emotion Recognition

A modern, beautiful web application for real-time speech emotion recognition using deep learning. Upload audio files and instantly detect emotions with confidence scores and detailed predictions.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 📋 Table of Contents

- [Features](#-features)
- [Supported Emotions](#-supported-emotions)
- [Screenshots](#-screenshots)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)
- [Author](#-author)
- [License](#-license)

## ✨ Features

- 🎨 **Modern UI**: Beautiful, responsive interface with smooth animations and gradient designs
- 🎤 **Multi-format Support**: Upload WAV, MP3, M4A, or FLAC audio files
- 🔮 **Real-time Prediction**: Instant emotion detection with AI-powered analysis
- 📊 **Detailed Analytics**: View top 3 predictions with confidence scores and visual progress bars
- 🎯 **High Accuracy**: Deep learning model trained on multiple emotion datasets
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices
- ⚡ **Fast Processing**: Optimized feature extraction and model inference
- 🎭 **8 Emotion Classes**: Detect angry, calm, disgust, fearful, happy, neutral, sad, and surprised

## 😊 Supported Emotions

The application can detect the following emotions:

| Emotion | Emoji | Description |
|---------|-------|-------------|
| Angry | 😠 | Frustrated or irritated speech |
| Calm | 😌 | Peaceful and relaxed tone |
| Disgust | 🤢 | Repulsed or revolted expression |
| Fearful | 😨 | Scared or anxious voice |
| Happy | 😊 | Joyful and cheerful speech |
| Neutral | 😐 | Normal, unemotional tone |
| Sad | 😢 | Melancholic or sorrowful voice |
| Surprised | 😲 | Astonished or amazed expression |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare model files**
   
   Make sure you have the trained model file in the project directory:
   - `best_model.keras` (preferred)
   - `best_emotion_model.keras`
   - `best_emotion_model_v2.keras`
   - `best_model_v2.keras`
   
   Also ensure you have the preprocessing files:
   - `scaler.pkl`
   - `label_encoder.pkl`

4. **Run the application**
   ```bash
   streamlit run app.py
   ```
   
   Or use the provided scripts:
   ```bash
   # Windows
   run_app.bat
   
   # Linux/Mac
   bash run_app.sh
   ```

5. **Open in browser**
   
   The app will automatically open at `http://localhost:8501`

## 📖 Usage

1. **Upload Audio**: Click "Browse files" or drag and drop an audio file
2. **Preview**: Listen to your audio using the built-in player (optional)
3. **Analyze**: Click the "🔮 Predict Emotion" button
4. **View Results**: 
   - See the detected emotion with emoji
   - Check confidence percentage
   - Review top 3 predictions with detailed scores
   - View confidence level indicators

### Tips for Best Results

- Use clear audio recordings without background noise
- WAV format at 22050 Hz sample rate recommended
- Audio length: 1-5 seconds works best (longer files are automatically truncated)
- Speak clearly and naturally

## 📁 Project Structure

```
project/
├── app.py                      # Main Streamlit web application
├── colab_training.py           # Training script for Google Colab
├── deep_project.py             # Local training script
├── deep_with_resultat_64.py    # Alternative training script
├── emotion_predictor.py        # Standalone prediction module
├── fix_numpy_pickle.py         # Utility for preprocessing fixes
├── save_preprocessing.py       # Helper to save preprocessing objects
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── run_app.bat                 # Windows launcher script
├── run_app.sh                  # Linux/Mac launcher script
├── best_model.keras            # Trained model (not included in repo)
├── scaler.pkl                  # Scaler object (not included in repo)
└── label_encoder.pkl           # Label encoder (not included in repo)
```

## 🔧 Technical Details

### Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Deep Learning**: TensorFlow/Keras
- **Audio Processing**: Librosa
- **Data Processing**: NumPy, scikit-learn
- **UI Styling**: Custom CSS with Google Fonts (Poppins)

### Model Architecture

The application uses a 1D Convolutional Neural Network (CNN) with:
- **Input Features**: 193 features extracted from audio
  - 40 MFCC (Mel-frequency Cepstral Coefficients)
  - 12 Chroma features
  - 128 Mel Spectrogram features
  - 7 Spectral Contrast features
  - 6 Tonnetz features

### Feature Extraction Pipeline

1. Audio loading at 22050 Hz sample rate
2. Duration: 3 seconds (with 0.5s offset)
3. Feature extraction using Librosa
4. Feature normalization using StandardScaler
5. Reshaping for model input

### Model Training

The model can be trained using:
- `colab_training.py` - For Google Colab environment
- `deep_project.py` - For local training

Training datasets used:
- RAVDESS (Ryerson Audio-Visual Database)
- CREMA-D (Crowd-sourced Emotional Multimodal Actors)
- TESS (Toronto Emotional Speech Set)
- SAVEE (Surrey Audio-Visual Expressed Emotion)

## 🐛 Troubleshooting

### Model Not Found

**Error**: "Model file not found"

**Solution**:
- Ensure your trained model file is in the project root directory
- Check the filename matches one of the supported names
- Verify the file size (should be ~25-100 MB)

### Preprocessing Files Missing

**Error**: Warnings about preprocessing files

**Solution**:
- Run `save_preprocessing.py` after training your model
- Or manually save scaler and label_encoder using pickle in your training script

### Audio Processing Issues

**Error**: "Error extracting features"

**Solution**:
- Verify audio file format is supported (WAV, MP3, M4A, FLAC)
- Check audio file is not corrupted
- Ensure audio has minimum length (~1 second)

### Import Errors

**Error**: Module not found

**Solution**:
```bash
pip install -r requirements.txt
```

### Port Already in Use

**Error**: Port 8501 already in use

**Solution**:
```bash
streamlit run app.py --server.port 8502
```

## 🎨 Customization

### Modify UI Colors

Edit the CSS in `app.py` to change colors:
- Primary gradient: `#667eea` to `#764ba2`
- Success banner: `#56ab2f` to `#a8e063`

### Adjust Model Paths

Modify the `possible_paths` list in `load_model()` function to add custom model filenames.

### Change Confidence Thresholds

Edit the confidence level checks in the results section:
- High: ≥ 70%
- Moderate: ≥ 50%
- Low: < 50%

## 👤 Author

**Fakher Ben Yahya**

- Project: Speech Emotion Recognition
- Built with ❤️ using Streamlit and TensorFlow

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- TensorFlow/Keras for deep learning capabilities
- Librosa for audio processing
- All the dataset creators (RAVDESS, CREMA-D, TESS, SAVEE)

---

**Made with ❤️ for emotion recognition research**

For issues or contributions, please feel free to open an issue or submit a pull request.
