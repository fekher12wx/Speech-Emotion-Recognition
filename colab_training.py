# ============================================
# SPEECH EMOTION RECOGNITION - GOOGLE COLAB
# Optimized for Colab with Free GPU
# ============================================

# Install required packages
print("📦 Installing packages...")
!pip install -q librosa kaggle

# ============================================
# MOUNT GOOGLE DRIVE & SETUP KAGGLE
# ============================================
from google.colab import drive
import os
import shutil

print("\n📂 Mounting Google Drive...")
drive.mount('/content/drive')

# Setup Kaggle API
print("\n🔑 Setting up Kaggle API...")
# Upload your kaggle.json to Colab files, then run:
# !mkdir -p ~/.kaggle
# !cp /content/kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# Or use kaggle.json from Drive:
KAGGLE_JSON_PATH = "/content/drive/MyDrive/kaggle.json"  # Update this path
if os.path.exists(KAGGLE_JSON_PATH):
    !mkdir -p ~/.kaggle
    shutil.copy(KAGGLE_JSON_PATH, '/root/.kaggle/kaggle.json')
    !chmod 600 ~/.kaggle/kaggle.json
    print("✅ Kaggle API configured!")
else:
    print("⚠️ kaggle.json not found. Please upload it to Colab or Drive.")

# ============================================
# IMPORT LIBRARIES
# ============================================
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Add
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler, CSVLogger
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import pickle
import time

print("✅ All libraries imported!")

# ============================================
# GPU CHECK
# ============================================
print("\n🖥️ Checking GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Available: {gpus[0].name}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("🚀 Training will use GPU (much faster!)")
else:
    print("⚠️ No GPU - using CPU (will be slower)")

# Enable mixed precision for GPU
if gpus:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("✅ Mixed precision enabled")

# ============================================
# DOWNLOAD DATASETS
# ============================================
print("\n📦 Downloading datasets from Kaggle...")
!mkdir -p /content/data

datasets = [
    ("uwrfkaggler/ravdess-emotional-speech-audio", "RAVDESS"),
    ("ejlok1/cremad", "CREMA-D"),
    ("ejlok1/toronto-emotional-speech-set-tess", "TESS"),
    ("ejlok1/surrey-audiovisual-expressed-emotion-savee", "SAVEE")
]

for dataset, name in datasets:
    print(f"\n📥 Downloading {name}...")
    !kaggle datasets download -d {dataset} -p /content/data --unzip -q
    print(f"✅ {name} downloaded")

print("\n✅ All datasets downloaded!")

# ============================================
# ORGANIZE DATASETS
# ============================================
print("\n📁 Organizing datasets...")
!mkdir -p /content/organized_data/Ravdess
!mkdir -p /content/organized_data/Crema
!mkdir -p /content/organized_data/Tess
!mkdir -p /content/organized_data/Savee

def organize_datasets():
    base_path = "/content/data/"
    organized_path = "/content/organized_data/"
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                
                if 'Actor_' in root or file.startswith('03-'):
                    dest = os.path.join(organized_path, 'Ravdess', file)
                    if not os.path.exists(dest):
                        shutil.copy2(file_path, dest)
                elif '_' in file and any(x in file for x in ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']):
                    dest = os.path.join(organized_path, 'Crema', file)
                    if not os.path.exists(dest):
                        shutil.copy2(file_path, dest)
                elif any(emotion in file.lower() for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad', 'pleasant']):
                    dest = os.path.join(organized_path, 'Tess', file)
                    if not os.path.exists(dest):
                        shutil.copy2(file_path, dest)
                elif file[0].lower() in ['a', 'd', 'f', 'h', 'n', 's'] and len(file) < 15:
                    dest = os.path.join(organized_path, 'Savee', file)
                    if not os.path.exists(dest):
                        shutil.copy2(file_path, dest)

organize_datasets()

# Count files
for dataset in ['Ravdess', 'Crema', 'Tess', 'Savee']:
    path = f"/content/organized_data/{dataset}"
    count = len([f for f in os.listdir(path) if f.endswith('.wav')]) if os.path.exists(path) else 0
    print(f"  {dataset:10s}: {count:4d} files")

# ============================================
# EMOTION EXTRACTION FUNCTIONS
# ============================================
def extract_emotion_ravdess(filename):
    emotion_dict = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    try:
        emotion_code = filename.split('-')[2]
        return emotion_dict.get(emotion_code, 'unknown')
    except:
        return 'unknown'

def extract_emotion_crema(filename):
    emotion_dict = {
        'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful',
        'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
    }
    try:
        emotion_code = filename.split('_')[2]
        return emotion_dict.get(emotion_code, 'unknown')
    except:
        return 'unknown'

def extract_emotion_tess(filename):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
    filename_lower = filename.lower()
    for emotion in emotions:
        if emotion in filename_lower:
            return 'surprised' if emotion == 'ps' else emotion
    return 'unknown'

def extract_emotion_savee(filename):
    emotion_dict = {
        'a': 'angry', 'd': 'disgust', 'f': 'fearful',
        'h': 'happy', 'n': 'neutral', 'sa': 'sad', 'su': 'surprised'
    }
    try:
        emotion_code = filename[0:2] if len(filename) > 1 and filename[1].isalpha() else filename[0]
        return emotion_dict.get(emotion_code.lower(), 'unknown')
    except:
        return 'unknown'

# ============================================
# ENHANCED FEATURE EXTRACTION
# ============================================
def extract_features(file_path, duration=3, offset=0.5):
    """Enhanced feature extraction"""
    try:
        audio, sr = librosa.load(file_path, duration=duration, offset=offset, sr=22050)

        # MFCC + Deltas
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta_mean = np.mean(mfccs_delta.T, axis=0)
        mfccs_delta_std = np.std(mfccs_delta.T, axis=0)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
        mfccs_delta2_mean = np.mean(mfccs_delta2.T, axis=0)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)
        chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)
        chroma_cens_mean = np.mean(chroma_cens.T, axis=0)

        # Mel
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_mean = np.mean(mel.T, axis=0)
        mel_std = np.std(mel.T, axis=0)
        mel_log = librosa.power_to_db(mel)
        mel_log_mean = np.mean(mel_log.T, axis=0)

        # Contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        contrast_std = np.std(contrast.T, axis=0)

        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)

        # Poly features
        try:
            poly_features = librosa.feature.poly_features(y=audio, sr=sr)
            poly_mean = np.mean(poly_features.T, axis=0)
            if poly_mean.ndim > 1:
                poly_mean = poly_mean.flatten()
        except:
            poly_mean = np.zeros(2)

        # Prosodic features
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        spectral_centroid_std = float(np.std(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
        spectral_rolloff_std = float(np.std(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
        spectral_bandwidth_std = float(np.std(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        zcr_std = float(np.std(librosa.feature.zero_crossing_rate(audio)))
        rms = float(np.mean(librosa.feature.rms(y=audio)))
        rms_std = float(np.std(librosa.feature.rms(y=audio)))

        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            tempo = float(tempo) if tempo > 0 else 0.0
        except:
            tempo = 0.0

        # Harmonic/Percussive
        try:
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_energy = float(np.mean(harmonic ** 2))
            percussive_energy = float(np.mean(percussive ** 2))
        except:
            harmonic_energy = 0.0
            percussive_energy = 0.0

        # Ensure all arrays are 1D
        features_list = [
            mfccs_mean, mfccs_std, mfccs_delta_mean, mfccs_delta_std, mfccs_delta2_mean,
            chroma_mean, chroma_std, chroma_cens_mean,
            mel_mean, mel_std, mel_log_mean,
            contrast_mean, contrast_std,
            tonnetz_mean, poly_mean,
            np.array([spectral_centroid, spectral_centroid_std, spectral_rolloff, spectral_rolloff_std,
                     spectral_bandwidth, spectral_bandwidth_std, zcr, zcr_std, rms, rms_std,
                     harmonic_energy, percussive_energy]),
            np.array([tempo])
        ]
        
        features_1d = []
        for feat in features_list:
            if isinstance(feat, (int, float)):
                features_1d.append(np.array([feat]))
            else:
                features_1d.append(feat.flatten())
        
        return np.concatenate(features_1d)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# ============================================
# LOAD DATASETS
# ============================================
def load_dataset(data_path, dataset_name, emotion_extractor):
    X, y = [], []
    file_count = 0
    print(f"\n📂 Loading {dataset_name}...")
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                emotion = emotion_extractor(file)
                
                if emotion != 'unknown':
                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(emotion)
                        file_count += 1
                        if file_count % 100 == 0:
                            print(f"  ✓ {file_count} files processed...")
    
    print(f"✅ {dataset_name}: {file_count} files")
    return X, y

print("\n" + "="*70)
print("📊 LOADING ALL DATASETS")
print("="*70)

X_all, y_all = [], []

for dataset_name, data_path, extractor in [
    ("RAVDESS", "/content/organized_data/Ravdess", extract_emotion_ravdess),
    ("CREMA-D", "/content/organized_data/Crema", extract_emotion_crema),
    ("TESS", "/content/organized_data/Tess", extract_emotion_tess),
    ("SAVEE", "/content/organized_data/Savee", extract_emotion_savee)
]:
    if os.path.exists(data_path):
        X, y = load_dataset(data_path, dataset_name, extractor)
        X_all.extend(X)
        y_all.extend(y)

X_all = np.array(X_all)
y_all = np.array(y_all)

print(f"\n✅ TOTAL SAMPLES: {len(X_all)}")
print(f"📏 Features: {X_all.shape[1]}")

# ============================================
# PREPROCESSING
# ============================================
print("\n🔧 Preprocessing...")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_all)
y_categorical = to_categorical(y_encoded)

print(f"Classes: {label_encoder.classes_}")
print(f"Number of classes: {len(label_encoder.classes_)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
X_reshaped = np.expand_dims(X_scaled, axis=2)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_categorical,
    test_size=0.15,
    random_state=42,
    stratify=y_categorical,
    shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train,
    shuffle=True
)

print(f"\n✅ Train: {X_train.shape}")
print(f"✅ Validation: {X_val.shape}")
print(f"✅ Test: {X_test.shape}")

# ============================================
# DATA AUGMENTATION
# ============================================
print("\n🔄 Augmenting training data...")

def augment_audio(X, y, augment_factor=2):
    X_aug, y_aug = [], []
    
    for i in range(len(X)):
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        for aug_idx in range(augment_factor - 1):
            sample = X[i].copy()
            noise_std = np.random.uniform(0.003, 0.008)
            noise = np.random.normal(0, noise_std, sample.shape)
            sample = sample + noise
            shift = np.random.randint(-30, 30)
            sample = np.roll(sample, shift, axis=0)
            
            if aug_idx == 1:
                scale_factor = np.random.uniform(0.9, 1.1)
                sample = sample * scale_factor
            
            X_aug.append(sample)
            y_aug.append(y[i])
    
    return np.array(X_aug), np.array(y_aug)

X_train_aug, y_train_aug = augment_audio(X_train, y_train, augment_factor=2)
print(f"Original: {X_train.shape}, Augmented: {X_train_aug.shape}")

# ============================================
# CREATE MODEL
# ============================================
print("\n🏗️ Creating model...")

def create_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Initial block
    x = Conv1D(64, 7, padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)
    
    # Residual block 1
    residual = x
    x = Conv1D(128, 5, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, 5, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    if residual.shape[-1] != x.shape[-1]:
        residual = Conv1D(128, 1, padding='same')(residual)
    x = Add()([x, residual])
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    # Residual block 2
    residual = x
    x = Conv1D(256, 3, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    if residual.shape[-1] != x.shape[-1]:
        residual = Conv1D(256, 1, padding='same')(residual)
    x = Add()([x, residual])
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.35)(x)
    
    # Residual block 3
    residual = x
    x = Conv1D(512, 3, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(512, 3, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    if residual.shape[-1] != x.shape[-1]:
        residual = Conv1D(512, 1, padding='same')(residual)
    x = Add()([x, residual])
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.4)(x)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    dense1 = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    
    dense2 = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.4)(dense2)
    
    dense3 = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(dense2)
    dense3 = BatchNormalization()(dense3)
    dense3 = Dropout(0.3)(dense3)
    
    outputs = Dense(num_classes, activation='softmax')(dense3)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = create_model(input_shape=(X_reshaped.shape[1], 1), num_classes=len(label_encoder.classes_))
model.summary()

# ============================================
# CALLBACKS
# ============================================
checkpoint = ModelCheckpoint(
    '/content/best_emotion_model.keras',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=5,
    min_lr=1e-7,
    verbose=0
)

def cosine_annealing(epoch, lr):
    if epoch < 5:
        return lr
    return lr * (0.5 * (1 + np.cos(np.pi * (epoch - 5) / 75)))

lr_scheduler = LearningRateScheduler(cosine_annealing, verbose=0)

callbacks = [checkpoint, early_stop, reduce_lr, lr_scheduler]

# ============================================
# CLASS WEIGHTS
# ============================================
y_train_labels = np.argmax(y_train_aug, axis=1)
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weight_dict = dict(enumerate(class_weights))
print("\nClass weights:", class_weight_dict)

# ============================================
# COMPILE & TRAIN
# ============================================
def top_3_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy', top_3_accuracy]
)

print("\n" + "="*70)
print("🚀 STARTING TRAINING")
print("="*70)
print(f"Training samples: {X_train_aug.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Epochs: 80 (with early stopping)")
if gpus:
    print(f"Device: GPU ⚡")
    print("Expected time: 20-40 minutes")
else:
    print(f"Device: CPU")
    print("Expected time: 1.5-3 hours")
print("="*70)

start_time = time.time()

history = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time

# ============================================
# EVALUATE
# ============================================
print("\n📈 Loading best model...")
model = keras.models.load_model('/content/best_emotion_model.keras')

print("\n📊 Evaluating on test set...")
test_loss, test_accuracy, test_top3 = model.evaluate(X_test, y_test, batch_size=64, verbose=1)
print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"✅ Test Top-3 Accuracy: {test_top3 * 100:.2f}%")

y_pred = model.predict(X_test, batch_size=64)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred_classes)
print(f"\n🎯 Final Accuracy: {accuracy * 100:.2f}%")

print("\n" + "="*70)
print("Classification Report")
print("="*70)
print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))

# ============================================
# SAVE FILES
# ============================================
print("\n💾 Saving model and preprocessing objects...")

# Save preprocessing
with open('/content/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('/content/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Files saved to /content/")
print("  • best_emotion_model.keras")
print("  • scaler.pkl")
print("  • label_encoder.pkl")

# ============================================
# SAVE TO GOOGLE DRIVE (BACKUP)
# ============================================
print("\n💾 Saving to Google Drive...")
DRIVE_SAVE_PATH = "/content/drive/MyDrive/emotion_model/"
os.makedirs(DRIVE_SAVE_PATH, exist_ok=True)

shutil.copy('/content/best_emotion_model.keras', DRIVE_SAVE_PATH + 'best_emotion_model.keras')
shutil.copy('/content/scaler.pkl', DRIVE_SAVE_PATH + 'scaler.pkl')
shutil.copy('/content/label_encoder.pkl', DRIVE_SAVE_PATH + 'label_encoder.pkl')

print(f"✅ Files backed up to Drive: {DRIVE_SAVE_PATH}")

# ============================================
# DOWNLOAD FILES
# ============================================
from google.colab import files

print("\n📥 Downloading files to your computer...")
files.download('/content/best_emotion_model.keras')
files.download('/content/scaler.pkl')
files.download('/content/label_encoder.pkl')

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"⏱️  Training time: {training_time/60:.2f} minutes")
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
print("="*70)
print("\n🎉 SUCCESS! Download the files and use them in your web app!")

