# GenderEmotionRecognizer – Reconnaissance du Genre et de l’Émotion à partir de la Parole  
> **Dépôt GitHub** : [https://github.com/MamiyaMegueye/GenderEmotionRecognizer/tree/main](https://github.com/MamiyaMegueye/GenderEmotionRecognizer/tree/main)

## Introduction  
Ce projet vise à prédire le **genre** et l'**émotion** à partir d'enregistrements audio de parole. Nous avons utilisé diverses techniques de **machine learning** et de **deep learning** pour classifier les caractéristiques audio extraites d’un jeu de données reconnu.  

## Jeu de Données  
Nous avons utilisé la base de données **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)** :  
- **Auteurs** : Livingstone & Russo  
- **Licence** : **CC BY-NA-SC 4.0**  
- **Source** : [Zenodo - RAVDESS Dataset](https://zenodo.org/record/1188976)  
- **Fichiers utilisés** : **Audio_Speech_Actors_01-24**  

## Extraction des Caractéristiques  
Pour analyser les fichiers audio, nous avons extrait plusieurs caractéristiques acoustiques :  
- **MFCCs (Mel-Frequency Cepstral Coefficients)** : `mfcc0` à `mfcc15`  
- **Taux de passage par zéro (Zero-Crossing Rate - ZCR)**  
- **Énergie RMS (Root Mean Square - RMS)**  
- **Centroïde spectral**  
- **Largeur de bande spectrale**  

## Prétraitement des Données  
- **Normalisation** : Standardisation des valeurs des caractéristiques  
- **Équilibrage** : Répartition équilibrée des genres et des émotions  

## Modèles de Machine Learning  
Nous avons testé différents modèles de machine learning et obtenu les précisions suivantes :  

| Modèle              | Précision |
|---------------------|----------|
| Régression Logistique | 33.42% |
| Forêt Aléatoire (Random Forest) | 76.01% |
| XGBoost | 63.00% |

## Modèle de Deep Learning (LSTM)  
Afin d'améliorer les performances, nous avons développé un **réseau de neurones basé sur LSTM**, atteignant :  
**Précision : 94.07%**  
**Perte (Loss) : 0.2632**  

## Entraînement du Modèle  
Le modèle LSTM a été entraîné avec les paramètres suivants :  
- **Nombre d’époques** : 100  
- **Taille du batch** : 32  
- **Optimiseur** : Adam  
- **Fonction de perte** : Categorical Crossentropy  

## Déploiement  
Le projet contient tous les fichiers nécessaires pour déployer le modèle. Voici la structure du dossier :  

projet/  
├── **app.py**             
├── **lstm_model.h5**     
├── **scaler.pkl**           
├── **requirements.txt**   
├── **templates/**        
│   └── **index.html**     
├── **ProjetFinal.ipynb** 
├── **README.md** 


### Description des fichiers :
- **`app.py`** : Contient le code backend pour charger le modèle et gérer les requêtes API.  
- **`lstm_model.h5`** : Modèle LSTM pré-entraîné utilisé pour la prédiction.  
- **`scaler.pkl`** : Fichier permettant de normaliser les données d'entrée avant la prédiction.  
- **`requirements.txt`** : Liste des dépendances Python requises pour exécuter l'application.  
- **`templates/index.html`** : Interface frontend pour interagir avec le modèle via une application web.
- **`ProjetFinal.ipynb`**  : Notebook Jupyter avec le code de préparation des données, l'entraînement du modèle et l'évaluation
- **`README.md`**          :Fichier contenant les instructions pour le déploiement et l'utilisation du projet

