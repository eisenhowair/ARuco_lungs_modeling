import cv2
import matplotlib.pyplot as plt

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

id = 2  # C'est l'identifiant du marqueur, tu peux le changer selon tes besoins
img_size = 700  # Définis la taille de l'image finale
marker_img = cv2.aruco.generateImageMarker(aruco_dict, id, img_size)

# Utilise cv2.imshow() pour afficher l'image (matplotlib peut poser problème avec les images générées par OpenCV)
cv2.imshow("Aruco Marker", marker_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Enregistre l'image
cv2.imwrite("aruco{}.png".format(id), marker_img)
