import os
import tempfile
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import numpy as np
import librosa
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import tensorflow as tf
import streamlit as st
import librosa.display

from functional_build_model_no_addons import build_model

def preprocess_audio(file_path, sr=48000):
    y, _ = librosa.load(file_path, sr=sr)
    y = scipy.signal.wiener(y)
    energy = np.array([np.sum(np.abs(y[i:i+1024]**2)) for i in range(0, len(y), 512)])
    threshold = np.percentile(energy, 80)
    voiced_indices = energy > threshold
    voiced_signal = np.concatenate([y[i*512:(i+1)*512] for i in range(len(voiced_indices)) if voiced_indices[i]])
    return voiced_signal, sr

def extract_formants(y, sr):
    N = 2048
    y = y[:N] * np.hamming(N)
    A = librosa.lpc(y, order=10)
    roots = np.roots(A)
    roots = roots[np.imag(roots) >= 0]
    angz = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angz * (sr / (2 * np.pi))
    formants = sorted(freqs[:4]) if len(freqs) >= 4 else [0, 0, 0, 0]
    return formants

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    f0 = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y).mean()
    phase = np.angle(librosa.stft(y))
    phase_diff_std = np.std(np.diff(phase, axis=1))
    formants = extract_formants(y, sr)
    features = np.vstack([mfcc, chroma, contrast])
    features = StandardScaler().fit_transform(features.T)
    spoof_features = np.array([spectral_flatness, phase_diff_std] + formants)
    return features, spoof_features

def load_dataset_from_wavs(folder="dataset"):
    X, y, spoof_feats = [], [], []
    for fname in os.listdir(folder):
        if fname.endswith(".wav"):
            label = 1 if "real" in fname else 0
            y.append(label)
            y_signal, sr = preprocess_audio(os.path.join(folder, fname))
            feat, spoof = extract_features(y_signal, sr)
            if feat.shape[0] < 100:
                feat = np.pad(feat, ((0, 100 - feat.shape[0]), (0, 0)), mode='constant')
            elif feat.shape[0] > 100:
                feat = feat[:100, :]
            X.append(np.expand_dims(feat, axis=-1))
            spoof_feats.append(spoof)
    return np.array(X), np.array(y), np.array(spoof_feats)

def cyclical_lr(epoch):
    base_lr, max_lr, step_size = 1e-4, 1e-3, 5
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, (1 - x))

def train_model():
    X, y, spoof_feats = load_dataset_from_wavs()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model = build_model(X.shape[1:])
    clr_callback = tf.keras.callbacks.LearningRateScheduler(cyclical_lr)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[clr_callback])
    model.save("vfd_model.h5")
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(spoof_feats, y)
    joblib.dump(rf, "rf_spoof.pkl")

def log_authentication(cnn, spoof, final, decision):
    log_file = "auth_log.xlsx"
    log_entry = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "CNN_Score": cnn,
        "Spoof_Score": spoof,
        "Final_Score": final,
        "Decision": decision
    }])
    if os.path.exists(log_file):
        old_log = pd.read_excel(log_file)
        full_log = pd.concat([old_log, log_entry], ignore_index=True)
    else:
        full_log = log_entry
    full_log.to_excel(log_file, index=False)

@st.cache_resource
def load_models():
    model = tf.keras.models.load_model("vfd_model.h5", compile=False)
    rf_model = joblib.load("rf_spoof.pkl")
    return model, rf_model

st.title("Voice Authentication (VFD)")
uploaded_file = st.file_uploader("Upload suara .wav", type=["wav"])

if st.button("Latih Ulang Model"):
    with st.spinner("Melatih model..."):
        train_model()
    st.success("Model berhasil dilatih dan disimpan.")

if uploaded_file:
    if uploaded_file.size > 5_000_000:
        st.error("File terlalu besar. Maksimum 5MB.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(tmp_path)
    y, sr = preprocess_audio(tmp_path)
    features, spoof_feat = extract_features(y, sr)
    input_data = tf.expand_dims(features, axis=0)
    input_data = tf.expand_dims(input_data, axis=-1)

    if not os.path.exists("vfd_model.h5") or not os.path.exists("rf_spoof.pkl"):
        st.warning("Model belum dilatih. Harap latih dulu.")
    else:
        model, rf_model = load_models()
        cnn_score = model.predict(input_data)[0][0]
        spoof_score = rf_model.predict_proba([spoof_feat])[0][1]
        final_score = 0.7 * cnn_score + 0.3 * spoof_score
        decision = "GRANTED" if final_score > 0.7 else "DENIED"
        st.write(f"CNN Score: {cnn_score:.2f}, Anti-Spoof Score: {spoof_score:.2f}")
        st.write(f"Final Authentication Score: {final_score:.2f}")

        if decision == "GRANTED":
            st.success("Access GRANTED")
        else:
            st.error("Access DENIED")

        log_authentication(cnn_score, spoof_score, final_score, decision)

        st.subheader("MFCC Visualization")
        fig, ax = plt.subplots()
        librosa.display.specshow(features[:, :, 0].T, sr=sr, x_axis='time')
        plt.title("MFCC")
        st.pyplot(fig)