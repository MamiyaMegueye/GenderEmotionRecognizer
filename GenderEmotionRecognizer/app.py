import os
import librosa
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

#Chargement des fichiers:
model = load_model('lstm_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

#Décodage des valeurs:
label_mapping = {
    0: 'female_angry', 1: 'female_calm', 2: 'female_disgust', 3: 'female_fearful', 4: 'female_happy', 
    5: 'female_neutral', 6: 'female_sad', 7: 'female_surprised', 8: 'male_angry', 9: 'male_calm', 
    10: 'male_disgust', 11: 'male_fearful', 12: 'male_happy', 13: 'male_neutral', 14: 'male_sad', 15: 'male_surprised'
}

#Fonction d'extraction des caractéristiques:
def extract_features(file_path, n_mfcc=15):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None, duration=2.5, offset=0.6)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        zcr = librosa.feature.zero_crossing_rate(y=audio).mean()
        rms = librosa.feature.rms(y=audio).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate).mean()
        return {
            "MFCCs": mfccs.T,
            "ZCR": zcr,
            "RMS": rms,
            "Spectral_Centroid": spectral_centroid,
            "Spectral_Bandwidth": spectral_bandwidth
        }
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques pour le fichier {file_path}: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    predicted_label = None
    file_name = "Aucun fichier" 

    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return render_template('index.html', file_name=file_name, prediction=predicted_label)

        audio_file = request.files['audio_file']
        if audio_file.filename == '':
            return render_template('index.html', file_name=file_name, prediction=predicted_label)

        file_name = audio_file.filename  
        file_path = "temp_audio.wav"
        audio_file.save(file_path)

        features = extract_features(file_path)
        if features is None:
            os.remove(file_path)
            return render_template('index.html', file_name=file_name, prediction="Erreur lors de l'extraction des features")

        mfccs = features["MFCCs"]
        frame = pd.DataFrame(mfccs, columns=[f'mfcc{x}' for x in range(mfccs.shape[1])])
        frame["ZCR"] = features["ZCR"]
        frame["RMS"] = features["RMS"]
        frame["Spectral_Centroid"] = features["Spectral_Centroid"]
        frame["Spectral_Bandwidth"] = features["Spectral_Bandwidth"]

        X = frame[['mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7',
                  'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14',
                  'ZCR', 'RMS', 'Spectral_Centroid', 'Spectral_Bandwidth']].values
        X = scaler.transform(X)
        X = np.expand_dims(X, axis=0)

        y_pred = model.predict(X)
        y_pred_labels = np.argmax(y_pred, axis=1)

        os.remove(file_path)

        prediction = y_pred_labels[0]
        predicted_label = label_mapping.get(prediction)

    return render_template('index.html', file_name=file_name, prediction=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
