import cv2

# Démarrer la capture vidéo
cap = cv2.VideoCapture(0)  # Utilise 0 pour la caméra par défaut, tu peux changer cela si nécessaire

# Nombre d'images à capturer
num_images = 20

# Répertoire pour enregistrer les images (assure-toi que ce répertoire existe)
output_directory = "images_calibration/"
image_counter = 0

while image_counter < num_images:
    # Lire une image de la caméra
    ret, frame = cap.read()

    # Afficher l'image en direct
    cv2.imshow('Capture Image', frame)

    # Attendre une touche, 'q' pour quitter, 'c' pour capturer l'image
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        image_path = output_directory + f"calibration_image_{image_counter}.png"
        cv2.imwrite(image_path, frame)
        print(f"Image {image_counter + 1} capturée: {image_path}")
        image_counter += 1

# Libérer la capture vidéo et fermer la fenêtre
cap.release()
cv2.destroyAllWindows()
