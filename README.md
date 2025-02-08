# GenderEmotionRecognizer
# 🎤 Gender & Emotion Recognition from Speech using Deep Learning

## 📖 Introduction
This project focuses on predicting **gender** and **emotion** from audio speech recordings. We used various **machine learning** and **deep learning** techniques to classify audio features extracted from a well-known dataset.

## 📂 Dataset
We used the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**:
- Authors: Livingstone & Russo  
- License: **CC BY-NA-SC 4.0**  
- Source: [Zenodo - RAVDESS Dataset](https://zenodo.org/record/1188976)  
- Files: **Audio_Speech_Actors_01-24**  

## 🎛️ Feature Extraction  
To analyze the audio files, we extracted key features:
- **MFCCs (Mel-Frequency Cepstral Coefficients)**: `mfcc0` to `mfcc18`
- **Zero-Crossing Rate (ZCR)**
- **Root Mean Square (RMS)**
- **Spectral Centroid**
- **Spectral Bandwidth**

## 🔧 Data Processing  
- **Normalization**: Standardizing feature values  
- **Balancing**: Ensuring equal distribution of gender and emotions  

## 🏆 Machine Learning Models  
We tested different ML models and obtained the following accuracies:

| Model           | Accuracy |
|----------------|----------|
| Logistic Regression | 33.42% |
| Random Forest       | 76.01% |
| XGBoost            | 63.00% |

## 🤖 Deep Learning Model (LSTM)  
To improve performance, we built an **LSTM-based neural network**, achieving:  
✅ **Accuracy: 94.07%**  
✅ **Loss: 0.2632**  

## 📌 Model Training  
The LSTM model was trained with:
- **100 epochs**
- **Batch size: 32**
- **Optimizer: Adam**
- **Loss function: Categorical Crossentropy**

## 📊 Results Visualization  
Loss and accuracy were plotted for both **training** and **validation** phases.

## 🚀 How to Use  
1. Clone this repository:  
   ```bash
   git clone https://github.com/YourUsername/YourRepository.git
   cd YourRepository

