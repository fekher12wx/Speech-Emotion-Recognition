# Speech Emotion Recognition Web Application

A web-based interface for Speech Emotion Recognition using Deep Learning. This application allows users to upload audio files and get real-time emotion predictions.

## Features

- 🎤 Upload audio files (WAV, MP3, M4A, FLAC)
- 🔮 Real-time emotion prediction
- 📊 Detailed confidence scores
- 🎨 Beautiful and intuitive web interface
- 📱 Responsive design

## Supported Emotions

- 😠 Angry
- 😌 Calm
- 🤢 Disgust
- 😨 Fearful
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprised

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Model

Make sure you have a trained model file in one of these formats:
- `best_emotion_model_v2.keras`
- `best_emotion_model.keras`

### 3. Save Preprocessing Objects (Important!)

After training your model, you need to save the `scaler` and `label_encoder` objects. Add these lines at the end of your training script (`deep_project.py`):

```python
from save_preprocessing import save_preprocessing_objects
save_preprocessing_objects(scaler, label_encoder)
```

This will create:
- `scaler.pkl` - StandardScaler object
- `label_encoder.pkl` - LabelEncoder object

**Note:** Without these files, the web app will use default preprocessing which may not work correctly with your trained model.

## Running the Application

### Start the Streamlit App

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. **Upload Audio File**: Click "Browse files" and select an audio file
2. **View Audio**: The audio player will display your uploaded file
3. **Predict Emotion**: Click the "🔮 Predict Emotion" button
4. **View Results**: See the detected emotion and confidence scores

## Project Structure

```
project/
├── app.py                  # Main Streamlit web application
├── emotion_predictor.py    # Prediction module
├── deep_project.py         # Training script (original)
├── save_preprocessing.py   # Helper to save preprocessing objects
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── best_emotion_model_v2.keras  # Trained model (you need to provide)
├── scaler.pkl             # Scaler object (generated after training)
└── label_encoder.pkl      # Label encoder (generated after training)
```

## Troubleshooting

### Model Not Found Error

If you see "Model file not found":
- Make sure your trained model file is in the project directory
- The model should be named `best_emotion_model_v2.keras` or `best_emotion_model.keras`

### Preprocessing Objects Not Found

If preprocessing objects are missing:
- Run the `save_preprocessing.py` script after training
- Or manually save them using pickle in your training script

### Audio Format Issues

- The app supports WAV, MP3, M4A, and FLAC formats
- For best results, use WAV files with 22050 Hz sample rate
- Very long audio files will be truncated to 3 seconds

## Technical Details

- **Framework**: Streamlit for web interface
- **Deep Learning**: TensorFlow/Keras
- **Audio Processing**: Librosa
- **Feature Extraction**: MFCC, Chroma, Mel Spectrogram, Spectral Contrast, Tonnetz
- **Model Architecture**: 1D CNN with Batch Normalization and Dropout

## Development

To modify the web interface, edit `app.py`. The main components are:
- Model loading (cached for performance)
- File upload handling
- Feature extraction
- Prediction and result display

## License

This project is for educational purposes.

## Support

For issues or questions, please check:
1. All dependencies are installed correctly
2. Model file exists and is compatible
3. Preprocessing objects are saved correctly
4. Audio file format is supported

---

Built with ❤️ using Streamlit and TensorFlow

