# Voice Frequency Detector App (VFD)

A Streamlit-based application for voice authentication using a CNN-BiLSTM model combined with anti-spoofing mechanisms via Random Forest.

## Features

- Voice pre-processing (denoising, voiced region extraction)
- Feature extraction: MFCC, Chroma, Spectral Contrast, Formants, Spectral Flatness, Phase
- CNN-based classification with anti-spoofing score
- Live authentication and access decision
- Logging to Excel
- Optional: Retrain model on your own dataset

## Installation

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run vfd_app_refactored.py
```

## File Structure

- `vfd_app_refactored.py`: Main application script
- `requirements.txt`: Python dependencies
- `functional_build_model_no_addons.py`: CNN-BiLSTM model builder
- `auth_log.xlsx`: Automatically generated after testing

## Notes

- Include `.wav` files in a folder named `dataset/` for training.
- Files named with "real" will be treated as real voice, others as spoofed.