import cv2
import numpy as np

def trouver_coordonnees_et_rayon(image, seuil_rouge=10):
    # Charger l'image

    # Convertir l'image de BGR à HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Définir la plage de teintes pour les rouges
    teinte_min = 0
    teinte_max = 20

    # Définir la plage de saturation pour les rouges
    saturation_min = 100
    saturation_max = 255

    # Définir la plage de valeur pour les rouges
    valeur_min = seuil_rouge
    valeur_max = 255

    # Créer un masque en utilisant les plages définies
    masque_rouge = cv2.inRange(image_hsv, (teinte_min, saturation_min, valeur_min), (teinte_max, saturation_max, valeur_max))

    # Trouver les contours des zones rouges
    contours, _ = cv2.findContours(masque_rouge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extraire les coordonnées du centre et du rayon des zones rouges
    coordonnees_et_rayons = []
    for contour in contours:
        # Trouver le centre et le rayon du cercle entourant le contour
        ((x, y), rayon) = cv2.minEnclosingCircle(contour)

        # Convertir les coordonnées en entiers
        x, y, rayon = int(x), int(y), int(rayon)

        # Exclure les cercles de rayon 0
        if rayon > 0:
            # Ajouter les coordonnées du centre et du rayon à la liste
            coordonnees_et_rayons.append(((x, y), int(rayon)))

    return coordonnees_et_rayons

def dessiner_cercles_sur_image(image_path, coordonnees_rayons, output_path="resultat_attention_map.jpg"):
    # Charger l'image
    image = cv2.imread(image_path)

    # Créer une copie de l'image pour dessiner les cercles
    image_cercles = np.copy(image)

    # Dessiner les cercles sur l'image
    for (x, y), rayon in coordonnees_rayons:
        # Dessiner le cercle
        cv2.circle(image_cercles, (x, y), rayon, (0, 255, 0), 2)

    # Enregistrer l'image avec les cercles dessinés
    cv2.imwrite(output_path, image_cercles)

def ecrire_variables_dans_fichier(variables, nom_fichier="coordonnees_cercles.py"):
    with open(nom_fichier, "w") as fichier:
        fichier.write("# Variables générées\n\n")
        for i, variable in enumerate(variables):
            fichier.write(f"variable_{i} = {variable}\n")

def normaliser_coordonnees_par_rapport_a_image(coordonnees_rayons, largeur_image, hauteur_image):
    coordonnees_normalisees = []
    dimension_maximale = max(largeur_image, hauteur_image)

    for (x, y), rayon in coordonnees_rayons:
        x_normalise =  -0.5 + (x / largeur_image)
        z_normalise = y / hauteur_image
        y_normalise =-0.4
        rayon_normalise = rayon / dimension_maximale
        coordonnees_normalisees.append((x_normalise, y_normalise, z_normalise, rayon_normalise))

    return coordonnees_normalisees


# Exemple d'utilisation
image_path = "attention_map3.jpeg"
image = cv2.imread(image_path)

hauteur_image, largeur_image, _ = image.shape
coordonnees_rayons = trouver_coordonnees_et_rayon(image)

coordonnees_rayons_normalisees = normaliser_coordonnees_par_rapport_a_image(coordonnees_rayons, largeur_image, hauteur_image)
# Afficher les coordonnées du centre et du rayon
for x,y,z, rayon in coordonnees_rayons_normalisees:
    print(f"Centre : {x,y,z}, Rayon : {rayon}")

# Dessiner les cercles sur l'image et enregistrer le résultat
dessiner_cercles_sur_image(image_path, coordonnees_rayons)
ecrire_variables_dans_fichier(coordonnees_rayons_normalisees)