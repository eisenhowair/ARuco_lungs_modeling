# Projet XYZ

## Description

Ce projet permet de modéliser un modèle de poumon avec infection en utilisant un modèle d'intelligence artificelle permettant de générer une carte d'attention à partir d'une image radiographique de poumon, des marqueurs ArUco, python et Blender.

## Programmes

### 1. aruco_generate.py

**Rôle :** Génère des marqueurs ArUco avec les paramètres spécifiés, y compris l'id et la dimension du marqueur.

### 2. aruco_capture.py

**Rôle :** Capture 20 images qui serviront pour la calibration et les place dans images_calibration.

### 2. recupere_cam_data.py

**Rôle :** Effectue la calibration du système de marqueurs ArUco, et en génère le fichier "Camera.npz".

### 2. (bis) aruco_calibrate.py

**Rôle :** Effectue la calibration, mais inscris les résultats dans le fichier "calibration_results.txt".

### 4. overlay_cam.py

**Rôle :** Programme Python qui génère des cartes d'attention à partir d'images radiographiques en utilisant un modèle d'IA entraîné.

### 5. reconnait_attention_map.py

**Rôle :**  Utilise la carte d'attention générée précédemment pour récupérer les coordonnées et dimensions des zones rouges présentes, signe d'infection, puis les inscrit dans "coordonnees_cercles.py".

### 5. displayer.py

**Rôle :**  Affiche le modèle 3D de poumon stocké dans "Models/Lung\_simplified" en utilisant "objloader2.py", avec les sphères dont les coordonnées et dimensions sont récupérées depuis "coordonnees_cercles.py".


## Auteur

[Elias Mouaheb]


