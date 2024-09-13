import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image
import cv2.aruco as aruco
from numpy import load
import numpy as np
import time
import math
from objloader2 import *
taille_modele = 0.4
frames_requises_entre_augmentations = 5  # Intervalle en secondes pour l'augmentation de la taille du modèle
size_increase_rate = 0.2  # Taux d'augmentation de la taille du modèle
last_generated_size = taille_modele  # Garde une trace de la dernière taille générée
frames_since_last_increase = 0
spheres_list = []





#################### FUNCTION TO ENABLE BASIC FUNCTIONS #############################

def init_gl():
    global obj
    glEnable(GL_DEPTH_TEST)                               #TO ENABLE DEPTH TESTING,SO THAT HIDDEN SURFACES ARE REMOVED
    glMatrixMode(GL_PROJECTION)  
    glEnable(GL_LIGHTING)  # Activer l'éclairage
    glEnable(GL_LIGHT0)    # Utiliser la lumière 0                         #TO ENABLE PROJECTION MATRIX TO DEFINE A CLIPPING SPACE(THE SHAPE IS THAT OF A FRUSTUM)
    glLoadIdentity()                                      #TO LOAD IDENTITY MATRIX,THE PROJECTION MATRIX RESETS TO IDENTITY MATRIX SO THAT NEW PARAMETERS ARE NOT COMBINED WITH PREVIOUS ONES
    gluPerspective(45, 640.0 /480, 0.1, 1000)             #PERSPECTIVE VIEW GIVES US A VIEW IN WHICH FARTHER OBJECTS APPEARS SMALLER COMPARED TO NEARER OBJECTS
    glViewport(0, 0, 640, 480)                            #IT TELLS WHICH PART OF WINDOW WILL BE VISIBLE,ITS UNIT IS SCREEN PIXEL UNITS
    global texture_object, texture_background             #DEFINING NAME OF TEXTURES TO BE USED FOR TEAPOT AND BACKGROUND RESPECTIVELY
    texture_object = glGenTextures(1)                     # RETURNS NAME OF THE TEXTURE, WHICH IS STORED IN TEXTURE_OBJECT,1 DENOTE THE NO. OF TEXTURE NAME RETURNED
    texture_background=glGenTextures(1)                   # RETURNS NAME OF THE TEXTURE, WHICH IS STORED IN TEXTURE_BACKGROUND,1 DENOTE THE NO. OF TEXTURE NAME RETURNED
    #glEnable(GL_LIGHTING)                                 #TO ENABLE LIGHTING
    #glEnable(GL_LIGHT0)                                   #TO USE LIGHT0
    #glLightfv(GL_LIGHT0, GL_POSITION, (0, 20, 20, 0.0))   #POSITION OF LIGHT0
    #glLightfv(GL_LIGHT0, GL_AMBIENT, (1,1,1, 1.0))
    #glLightfv(GL_LIGHT0, GL_DIFFUSE, (1,1,1, 1.0))        #AMBIENT AND DIFFUSE PARAMETERS OF LIGHT0
    glActiveTexture(GL_TEXTURE0)                          #THE ACTIVE TEXTURE UNIT, BY DEFAULT IT IS THE ACTIVE TEXTURE UNIT
    #obj = OBJ('Models/Lung_v11/3dlung_blend.obj', swapyz=True)
    obj = OBJ('Models/Lung_simplified/simplified_lung6.obj', swapyz=True)

    #obj = OBJ('Models/Lung_v3/poumon_transparent.obj', swapyz=True)
    obj.generate()
    


####################### FUNCTION TO CHECK WHETHER ANY ARUCO MARKER HAS BEEN DETECTED IN THE IMAGE OR NOT #######################

def check_markers(img):
    ar_module=aruco.Dictionary_get(aruco.DICT_5X5_250)                      #TO SELECT AN ARUCO DICTIONARY
    detect=aruco.DetectorParameters_create()                                #INITIALIZING THE DETECTOR
    mc,mid,_=aruco.detectMarkers(img,ar_module,parameters=detect)           #TO GET ID AND CORNER POINTS OF DETECTED ARUCO MARKERS
    return mc                                                               #RETURNING CORNER POINTS AS A MEANS TO CHECK WHETHER ANY MARKER IS DETECTED OR NOT


###################### FUNCTION TO GET THE LIST OF ARUCO MARKERS DETECTED IN THE IMAGE #########################
###################### THE LIST CONTAINS ARUCO ID,COORDINATE OF ITS CENTRE,ROTATION VECTOR OF ITS CENTRE,TRANSLATION VECTOR OF ITS CENTRE,CORNER POINTS OF MARKER ##############

def detect_markers(img,mtx,dist):
    ar_module=aruco.Dictionary_get(aruco.DICT_5X5_250)                       #TO SELECT AN ARUCO DICTIONARY
    detect=aruco.DetectorParameters_create()                                 #INITIALIZING THE DETECTOR
    mc,mid,_=aruco.detectMarkers(img,ar_module,parameters=detect)            #TO GET ID AND CORNER POINTS OF DETECTED ARUCO MARKERS
    aruco_lst=[]                                                             #LIST TO ADD DETECTED ARUCO MARKERS' PROPERTIES

    ################### LOOP TO ADD PROPERTIES OF DETECTED MARKERS IN THE LIST ###################################

    for i in range(len(mid)):
        aruco_id=mid[i]                                                            #ARUCO ID IS STORED IN ARUCO_ID
        rvec,tvec,_=aruco.estimatePoseSingleMarkers(mc,100,mtx,dist)               #ROTATION AND TRANSLATION VECTORS OF MARKER'S CENTRE IS CALCULATED AND STORED IN RVEC AND TVEC
        x=(mc[i][0][0][0]+mc[i][0][1][0]+mc[i][0][2][0]+mc[i][0][3][0])/4          # X-COORDINATE OF MID POINT OF ARUCO IS FOUND
        y=(mc[i][0][0][1] + mc[i][0][1][1] + mc[i][0][2][1] + mc[i][0][3][1])/4    # Y-COORDINATE OF MID POINT OF ARUCO IS FOUND
        centre=(x,y)                                                               # CENTRE COORDINATES IS STORED IN A TUPLE
        tpl=(np.array([aruco_id]),centre,rvec[i],tvec[i],mc[i])                    #ALL THE PROPERTIES OF A DETECTED ARUCO IS COLLECTED AND STORED IN A TUPLE
        aruco_lst.append(tpl)                                                      #THE CREATED TUPLE IS ADDED IN THE LIST
    return aruco_lst                                                               #THE FINAL LIST IS RETURNED


####################### FUNCTION TO DRAW LINES AROUND THE ARUCO MARKERS ##############################
####################### THE ID OF THE MARKER WILL ALSO BE WRITTEN ON TOP OF IT #######################

def show_detected_markers(img,mtx,dist,aruco_lst):
    """
    for i in range(len(aruco_lst)):
        id=aruco_lst[i][0][0][0]                                                                                 #GETTING ID OF ARUCO
        corner1=tuple(aruco_lst[i][4][0][0])
        corner2 = tuple(aruco_lst[i][4][0][1])
        corner3 = tuple(aruco_lst[i][4][0][2])
        corner4 = tuple(aruco_lst[i][4][0][3])                                                                   #GETTING COORDINATES OF ALL THE FOUR CORNER POINTS OF THE DETECTED ARUCO
        pts=np.float32([[0,0,0]])
        rvec = aruco_lst[i][2]
        tvec = aruco_lst[i][3]
        imgpts, _ = cv2.projectPoints(pts, rvec, tvec, mtx, dist)
        centre=tuple(imgpts[0][0])                                                                               #TO GET COORDINATE OF MID POINT OF MARKER
        img = cv2.line(img, corner1, corner2, (0, 0, 255), 5)
        img = cv2.line(img, corner2, corner3, (0, 0, 255), 5)
        img = cv2.line(img, corner3, corner4, (0, 0, 255), 5)
        img = cv2.line(img, corner4, corner1, (0, 0, 255), 5)                                                     #TO DRAW A BOX AROUND THE MARKER
        img=cv2.putText(img,"id="+str(id),centre,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3,cv2.LINE_AA)              #TO WRITE ID OF THE ARUCO
    """
    return img                                                                                                    #RETURNING THE IMAGE WITH DRAWN BORDER AND ID

############################# FUNCTION TO CREATE TEXTURE FOR LUNG ######################################

def init_object_texture(text):
    glEnable(GL_TEXTURE_2D)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D,texture_object)                                      #BINDING TEXTURE OBJECT texture_object TO TEXTURE UNIT 0
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP)                        #IT SPECIFIES HOW TO WRAP TEXTURE ALONG X-AXIS,GL_CLAMP IS USED TO STOP WRAPPING TEXTURE AFTER RANGE OF X-AXIS IS COMPLETE
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)                      #IT SPECIFIES HOW TO WRAP TEXTURE ALONG Y-AXIS,GL_CLAMP IS USED TO STOP WRAPPING TEXTURE AFTER RANGE OF Y-AXIS IS COMPLETE
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)                 #APPLYING TEXTURE FILTER
    glEnable(GL_BLEND)

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Activer le mélange pour la transparence
    
    img=cv2.imread(text)                                                             #CONVERTING IMAGE IN ARRAY FORM (BGR FORMAT)
    img = Image.fromarray(img)                                                       #CONVERTING IMAGE FROM ARRAY FORM TO PILLOW IMAGE FORM
    width=img.size[0]                                                                #CALCULATING WIDTH OF IMAGE
    height=img.size[1]                                                               #CALCULATING HEIGHT OF IMAGE
    img=img.tobytes("raw", "BGRX")                                                   #CONVERTING IMAGE TO BYTES

    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,img)#CREATES A 2-D TEXTURE IMAGE
    #glDisable(GL_BLEND)



############################## FUNCTION TO DISPLAY TEAPOT ################################################

from OpenGL.GL import *

def overlay(aruco_lst):

    global spheres_list
    error = 0  # ERROR VALUE DETERMINED FROM HIT AND TRIAL
    rvecs = aruco_lst[0][2]
    tvecs = aruco_lst[0][3][0]
    rmtx = cv2.Rodrigues(rvecs)[0]  # CONVERTING ROTATION VECTOR INTO ROTATION MATRIX

    K = 101.13058796569  # K IS THE SCALING FACTOR FOR Z-TRANSLATION VECTOR

    z = tvecs[2] / K  # Z-TRANSLATION VECTOR OF OPENGL HAS BEEN CREATED BY USING SCALING FACTOR

    thetax = math.atan(abs(tvecs[2] / tvecs[0]))  # TAN(THETAX) = Z-TRANSLATION VECTOR/X-TRANSLATION VECTOR    IN OPENCV
    thetay = math.atan(abs(tvecs[2] / tvecs[1]))  # TAN(THETAY) = Z-TRANSLATION VECTOR/Y-TRANSLATION VECTOR    IN OPENCV

    abs(thetax)
    abs(thetay)  # FOR NOW, ONLY POSITIVE ANGLES ARE NEEDED
    if (z > 10 and z < 22):
        error = 2.5
    if (z >= 22):
        error = 5  # ERROR VALUE DETERMINED FROM HIT AND TRIAL
    x = (z - error) / (math.tan(thetax))
    y = (z - error) / (math.tan(thetay))  # FINDIND X AND Y TRANSLATION VECTORS OF OPENGL FORMAT BY USING PREVIOUSLY FOUND ANGLES

    if (tvecs[0] < 0):
        x = -x
    if (tvecs[1] < 0):
        y = -y

    view_matrix = np.array([[rmtx[0][0], rmtx[0][1], rmtx[0][2], x],  # VIEW MATRIX OF OPENCV FORMAT
                            [rmtx[1][0], rmtx[1][1], rmtx[1][2], y],
                            [rmtx[2][0], rmtx[2][1], rmtx[2][2], z],
                            [0.0, 0.0, 0.0, 1.0]])

    inverse_matrix = np.array([[1.0, 1.0, 1.0, 1.0],  # AS DIRECTION OF Y AND Z AXIS ARE OPPOSITE IN OPENCV AND OPENGL REVERSE THEM BY MULTIPLYING THEM BY -1
                               [-1.0, -1.0, -1.0, -1.0],  # X AXIS HAS THE SAME DIRECTION IN BOTH FORMATS
                               [-1.0, -1.0, -1.0, -1.0],
                               [1.0, 1.0, 1.0, 1.0]])

    view_matrix = view_matrix * inverse_matrix
    view_matrix = np.transpose(view_matrix)  # CONVERTING VIEW MATRIX FROM ROW MAJOR FORMAT(USED IN OPENCV) TO COLUMN MAJOR FORMAT(USED IN OPENGL)

    glPushMatrix()  # COPYING THE TOPMOST MATRIX OF MODELVIEW STACK WHICH IS THE IDENTITY MATRIX AND BECOMING ITSELF THE TOPMOST MATRIX
    glLoadMatrixd(view_matrix)  # MULTIPLYING THE IDENTITY MATRIX WITH OUR VIEW MATRIX

    ############################## DRAW SPHERES ################################

    import coordonnees_cercles
    if not spheres_list:
        spheres_list = [getattr(coordonnees_cercles, var) for var in dir(coordonnees_cercles) if var.startswith('variable_')]
    # Chaque élément de la liste est un tuple (x, y, z, rayon). x négatif = poumon droit, z = hauteur, y = profondeur
    #spheres_list = [(-1.0, -1.0, 1.0, 1.0), (1.0, -1.0, 1.0, 0.8)]  # Exemple de liste de sphères

    for sphere in spheres_list:
        draw_sphere(*sphere)

    ############################# DRAW LUNGS ##################################

    obj.render()
    glPopMatrix()  # DELETING THE TOPMOST MATRIX OF MODELVIEW STACK


def draw_sphere(x, y, z, radius):
    glPushMatrix()
    color_factor = (radius - 0) / (0.3 - 0)  # Normalisation du rayon entre 0 et 0.3
    color_factor = max(0, min(1, color_factor))  # S'assurer que la valeur est entre 0 et 1

    # Interpolation linéaire entre jaune et rouge
    r = 1.0 - color_factor  # Plus proche de 0.3, plus rouge
    g = color_factor       # Plus proche de 0, plus jaune
    b = 0.0

    glColor3f(r, g, b)
    glTranslatef(x, y, z)  # Translate to the position of the sphere
    glutSolidSphere(radius, 20, 20)  # Draw a solid sphere
    glPopMatrix()


############################# FUNCTION TO DRAW BACKGROUND IN OPENGL ################################################

def draw_background(frame):
    glEnable(GL_TEXTURE_2D)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_background)                              #BINDING TEXTURE OBJECT texture_background TO TEXTURE UNIT 0
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                    GL_CLAMP)                                                     #IT SPECIFIES HOW TO WRAP TEXTURE ALONG X-AXIS,GL_CLAMP IS USED TO STOP WRAPPING TEXTURE AFTER RANGE OF X-AXIS IS COMPLETE
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                    GL_CLAMP)                                                     #IT SPECIFIES HOW TO WRAP TEXTURE ALONG Y-AXIS,GL_CLAMP IS USED TO STOP WRAPPING TEXTURE AFTER RANGE OF Y-AXIS IS COMPLETE
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)              #APPLYING TEXTURE FILTER



    frame = Image.fromarray(frame)                                                 # CONVERTING IMAGE FROM ARRAY FORM TO PILLOW IMAGE FORM
    width = frame.size[0]                                                          # CALCULATING WIDTH OF IMAGE
    height = frame.size[1]                                                         # CALCULATING HEIGHT OF IMAGE
    frame = frame.tobytes("raw", "BGRX")                                           # CONVERTING IMAGE TO BYTES

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame) #CREATES A 2-D TEXTURE IMAGE

    ############### CREATING A RECTANGLE AND APPLYING TEXTURE ON IT,THE RATIO OF RECTANGLE IS MAINTAINED AT 4:3 ########

    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0);glVertex2f(-32.0, 24.0)
    glTexCoord2f(0.0, 1.0);glVertex2f(-32.0, -24.0)
    glTexCoord2f(1.0, 1.0);glVertex2f(32.0, -24.0)
    glTexCoord2f(1.0, 0.0);glVertex2f(32.0, 24.0)
    glEnd()

##################### Détecte le rouge pour zoom in ###########################
def count_red_pixels(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 130, 130])
    upper_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    red_pixels = cv2.countNonZero(red_mask)
    return red_pixels

def count_green_pixels(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    green_pixels = cv2.countNonZero(green_mask)
    return green_pixels


############################ THIS FUNCTION RUNS CONTINOUSLY TO DISPLAY THE SCENE IN OPENGL ############################

def drawGLScene():
    seuil_pixel_rouge = 10000
    seuil_pixel_vert = 10000
    global last_generated_size, frames_since_last_increase, spheres_list
    glClearColor(1,1,1,1)                               #TO SELECT COLOUR WHICH WILL BE DISPLAYED AFTER SCREEN IS CLEARED
    glClearDepth(1.0)                                              #TO SELECT VALUE OF DEPTH/Z BUFFER AFTER SCREEN IS CLEARED,1.0 MEANS THE FARTHEST
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)             #CLEARING COLOUR BUFFER AND DEPTH BUFFER


    ret, frame = cap.read()                                        #GETTING THE FRAME READ BY CAMERA



    if ret == True:
        mc = check_markers(frame)                                   # FUNCTION TO CHECK WHETHER A MARKER IS PRESENT IN THE IMAGE
        glPushMatrix()                                              # COPYING THE TOP MATRIX IN PROJECTION MATRIX STACK AND BECOMING ITSELF THE TOP MATRIX
        glTranslate(0,0,-58)                                      # TRANSLATING OUR BACKGROUND TO Z=-58 TO FIT ON THE SCREEN AND GIVING US A LARGER DEPTH OUR TEAPOT
        
        glDisable(GL_DEPTH_TEST)
        draw_background(frame)
        glEnable(GL_DEPTH_TEST)                                    
        
        glPopMatrix()                                               # DESTROYING THE TOP MATRIX OF PROJECTION MATRIX STACK



        ##################### IF ANY MARKER IS DETECTED IN THE IMAGE THEN BELOW ALGORITHM IS EXECUTED #################

        if (len(mc) != 0):
            glDisable(GL_LIGHTING)
            aruco_lst = detect_markers(frame, mtx, dist)             # FUNCTION TO RETURN A LIST OF ARUCO MARKERS AND THEIR PROPERTIES
            img = show_detected_markers(frame, mtx, dist, aruco_lst) # FUNCTION TO DRAW BORDER AROUND DETECTED MARKERS AND TO DISPLAY ITS ID
            glMatrixMode(GL_MODELVIEW)                               #LOADING MODELVIEW STACK
            glLoadIdentity()                                         #LOADING IDENTITY MATRIX
            ################ WHEN ARUCO ID=2 IS DETECTED ##################

            if aruco_lst[0][0][0][0] == 2 or aruco_lst[0][0][0][0] == 8:
                red_pixel_count = count_red_pixels(frame)  # Remplacez count_red_pixels par la fonction qui compte les pixels rouges
                print("rouge : " + str(red_pixel_count))
                green_pixel_count = count_green_pixels(frame)  # Remplacez count_red_pixels par la fonction qui compte les pixels rouges
                print("vert : " + str(green_pixel_count))

                if  red_pixel_count > seuil_pixel_rouge:
                    if frames_since_last_increase >= frames_requises_entre_augmentations:
                        last_generated_size += size_increase_rate  # Augmentation de la taille du modèle
                        obj.generate(last_generated_size)  # Génère le modèle avec la nouvelle taille
                        frames_since_last_increase = 0  # Réinitialise le compteur
                        update_spheres_list(True)
                        print("taille modèle:", last_generated_size)  # Ajout de cette ligne pour le débogage
                    else:
                        frames_since_last_increase +=1
                elif green_pixel_count > seuil_pixel_vert and last_generated_size > 0.4:
                    if frames_since_last_increase >= frames_requises_entre_augmentations:
                        last_generated_size -= size_increase_rate  # Augmentation de la taille du modèle
                        obj.generate(last_generated_size)  # Génère le modèle avec la nouvelle taille
                        frames_since_last_increase = 0  # Réinitialise le compteur
                        update_spheres_list(False)
                        print("taille modèle:", last_generated_size)  # Ajout de cette ligne pour le débogage
                    else:
                        frames_since_last_increase +=1

                overlay(aruco_lst)                                   # FUNCTION TO DISPLAY


        cv2.waitKey(1)                                               #GAP BETWEEN TWO FRAMES OF OPENCV IS 1 MILLI SECOND

    glutSwapBuffers()                                                #SWAPS THE FRONT AND BACK BUFFER

def update_spheres_list2(grow):

    global spheres_list
    coeff_x = 0.12 # axe largeur
    coeff_y = -0.1 # profondeur
    coeff_z = 0.12 # hauteur
    for i in range(len(spheres_list)):
        if grow: # cas où le modèle grandit
            if(spheres_list[i][0] > 0):
                spheres_list[i] = (spheres_list[i][0] + coeff_x, spheres_list[i][1]+ coeff_y, spheres_list[i][2]+ coeff_z, spheres_list[i][3] + coeff_z*0.7)
                pass
            else:
                spheres_list[i] = (spheres_list[i][0] - coeff_x, spheres_list[i][1]+ coeff_y, spheres_list[i][2]+ coeff_z, spheres_list[i][3] + coeff_z*0.7)
        else: # cas où le modèle rapetisse
            if(spheres_list[i][0] > 0):
                spheres_list[i] = (spheres_list[i][0] - coeff_x, spheres_list[i][1] - coeff_y, spheres_list[i][2] - coeff_z, spheres_list[i][3] - coeff_z*0.7)
                pass
            else:
                spheres_list[i] = (spheres_list[i][0] + coeff_x, spheres_list[i][1] - coeff_y, spheres_list[i][2] - coeff_z, spheres_list[i][3] - coeff_z*0.7)
            pass

def update_spheres_list(grow):

    global spheres_list
    coeff_x = 1.2 # axe largeur
    coeff_y = 1.15 # profondeur
    coeff_z = 1.2 # hauteur

    for i in range(len(spheres_list)):
        if grow: # cas où le modèle grandit
            spheres_list[i] = (spheres_list[i][0] * coeff_x, spheres_list[i][1] * coeff_y, spheres_list[i][2] * coeff_z, spheres_list[i][3] * (coeff_z))
            pass
        else: # cas où le modèle rapetisse
            spheres_list[i] = (spheres_list[i][0] / coeff_x, spheres_list[i][1] / coeff_y, spheres_list[i][2] / coeff_z, spheres_list[i][3] / (coeff_z))
            pass



data=load("aruco_setup/Camera.npz")                                              # LOADING THE FILE CONTAINING CAMERA PARAMETERS
dist=data["dist"]                                                    # TAKING DATA IN FILE NAME DIST
mtx=data["mtx"]                                                      # TAKING DATA IN FILE NAME MTX
cap=cv2.VideoCapture(0)                                              # TO OPEN OUR CAMERA
glutInit()                                                           # CREATES A GLUT OBJECT THAT ALLOWS US TO CUSTOMIZE OUR WINDOW
glutInitWindowSize(1400, 1050)                                       # WINDOW SIZE
glutInitWindowPosition(800, 600)                                     # WINDOW DISPLAY POSITION
glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH|GLUT_DOUBLE)              # DISPLAY COLOUR OF RGBA FORMAT,INCLUDING DEPTH BUFFER AND DOUBLE BUFFER IS ALSO SELECTED
glutCreateWindow("Fenetre modele")                                   #NAME OF WINDOW
init_gl()                                                            #FUNCTION TO INITIALISE BASIC FUNCTIONS
glutDisplayFunc(drawGLScene)                                         #IT RUNS THE FUNCTION DRAWGLSCENE WHEN WINDOW IS DISPLAYED
glutIdleFunc(drawGLScene)                                            #IT CONTINOUSLY RUNS THE DRAWGLSCENE FUNCTION
glutMainLoop()                                                       #THIS FUNCTION LOOKS AT THE EVENTS IN THE QUEUE,FOR EACH EVENT IN THE QUEUE IT EXECUTES A CALL BACK FUNCTION
                                                                     #IF NO CALL BACK FUNCTION IS DEFINED FOR THAT EVENT THEN THAT EVENT IS IGNORED