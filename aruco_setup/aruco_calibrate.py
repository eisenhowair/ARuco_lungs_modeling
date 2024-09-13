import cv2
import numpy as np
import os

# Chemin vers le répertoire contenant les images
directory_path = "images_calibration/"

# Liste pour stocker les chemins d'accès aux images
image_paths = []

# Parcourir tous les fichiers du répertoire
for filename in os.listdir(directory_path):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        # Vérifier que le fichier est une image (avec une extension png, jpg, ou jpeg)
        image_paths.append(os.path.join(directory_path, filename))


def calibrate_camera(images, pattern_size=(7, 7), marker_size=0.04):
    # Préparation des objets de points du monde réel (coordonnées 3D du marqueur)
    objp = np.array([[0, 0, 0], [marker_size, 0, 0], [0, marker_size, 0], [marker_size, marker_size, 0]], dtype=np.float32)

    # Tableaux pour stocker les objets 3D et les points 2D pour toutes les images
    objpoints = []  # Points du monde réel
    imgpoints = []  # Points de l'image (coins détectés)

    for image_path in images:
        # Charger l'image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Détecter le marqueur ArUco
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Si le marqueur est trouvé, ajoute les points correspondants aux tableaux
        if ids is not None:
            objpoints.append(objp)
            imgpoints.append(corners[0])  # On suppose qu'un seul marqueur est présent dans l'image

            # Dessine le marqueur sur l'image
            cv2.aruco.drawDetectedMarkers(img, corners, ids)

    # Calibration de la caméra
    ret, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    with open('calibration_results.txt', 'w') as file:
        file.write("Matrice de la caméra (cameraMatrix):\n")
        file.write(str(cameraMatrix.tolist()))
        file.write("\nCoefficients de distorsion (distCoeffs):\n")
        file.write(str(distCoeffs.tolist()))

    return cameraMatrix, distCoeffs

# Exemple d'utilisation
if __name__ == "__main__":

    # Appel de la fonction de calibration
    cameraMatrix, distCoeffs = calibrate_camera(image_paths)

    # Afficher les résultats
    print("Matrice de la caméra (cameraMatrix):\n", cameraMatrix)
    print("Coefficients de distorsion (distCoeffs):\n", distCoeffs)
