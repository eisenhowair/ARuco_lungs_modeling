import cv2
import numpy as np

# Charger le dictionnaire ArUco utilisé précédemment
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Initialiser le détecteur ArUco
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

with open("calibration_results.txt", "r") as file:
    lines = file.readlines()
    cameraMatrix = np.array(eval(lines[1]))  # Ignorer la première ligne
    distCoeffs = np.array(eval(lines[3]))  # Ignorer la troisième ligne


# Démarrer la capture vidéo
cap = cv2.VideoCapture(0)  # Utilise 0 pour la caméra par défaut, tu peux changer cela si nécessaire

while True:
    # Lire une image de la caméra
    ret, frame = cap.read()

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les marqueurs ArUco
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params
    )

    # Si des marqueurs sont détectés, dessine les contours et affiche l'ID
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, 1, cameraMatrix, distCoeffs
        )

        for i in range(len(ids)):
            # Affiche l'ID du marqueur
            cv2.putText(frame, str(ids[i][0]), tuple(map(int, corners[i][0][0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # Dessine le cube autour du marqueur
            axis_points = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]).reshape(-1, 3, 1)
            axis_points, _ = cv2.projectPoints(axis_points, rvecs[0], tvecs[i], cameraMatrix, distCoeffs)

            # Convertir les points en entiers
            axis_points = axis_points.squeeze().astype(int)

            # Dessiner les lignes représentant les axes
            cv2.line(frame, tuple(axis_points[0]), tuple(axis_points[1]), (0, 0, 255), 2)  # Axe X (rouge)
            cv2.line(frame, tuple(axis_points[0]), tuple(axis_points[2]), (0, 255, 0), 2)  # Axe Y (vert)
            cv2.line(frame, tuple(axis_points[0]), tuple(axis_points[2]), (255, 0, 0), 2)  # Axe Z (bleu)


        # Dessine les contours et les axes 3D (en dehors de la boucle for)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Afficher l'image
    cv2.imshow("ArUco Marker Detection", frame)

    # Sortir de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libérer la capture vidéo et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
