import cv2 as cv
import numpy as np
import subprocess
import yaml
import rosbag
from cv_bridge import CvBridge
import math 

def detect_aruco(nom_im = "image_exemple.png", nom_out = " ", ecritur = False) :
    if(ecritur):
        if(nom_out == " "):
            nom_out = "marker_detect_" + nom_im

    image = cv.imread(nom_im)

    #Load the dictionary that was used to generate the markers.
    dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_100,)

    # Initialize the detector parameters using default values
    parameters =  cv.aruco.DetectorParameters_create()

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(image, dictionary, parameters=parameters)

    #enregistrement dans une nouvelle image
    if(ecritur):
        outputImage = image.copy()
        cv.aruco.drawDetectedMarkers(outputImage, markerCorners, markerIds)
        cv.imwrite(nom_out, outputImage)    
    #print( "fin de detection d'aruco pour l'image ", nom_im)
    return(markerCorners, markerIds)

def find_pos_3d_to_2d(markerCorners, markerIds, id_origin, nom_im = "image_exemple.png", nom_out_axe = " ", 
    nom_out_detect = " ", obj_bleu = [[2,5,0],[1,3,0]] , obj_vert = [[5,5,0],[0,0,0]], ecritur = False):
    
    #generer les nom
    if(ecritur):
        if(nom_out_axe == " "):
            nom_out_axe = "axis_detect_" + nom_im

        if(nom_out_detect == " "):
            nom_out_detect = "points3D_2_2D" + nom_im


   

    image = cv.imread(nom_im)

    size_of_marker =  0.029 # side lenght of the marker in meter

    #trouvé le Id d'origine
    a = np.where(markerIds == id_origin)
    array_pos_origin = a[0][0]
    markerCorners.append(markerCorners[array_pos_origin])

    #valeur de caméra
    mtx_camera = np.array([[886.773967, 0.000000, 627.460303], 
        [0.000000, 888.130640, 367.062129],
        [0.000000, 0.000000, 1.000000]])

    distortion = np.array([0.095401, -0.178426, -0.001307, -0.010675, 0.0])

    #detection axe
    rvecs,tvecs, trash = cv.aruco.estimatePoseSingleMarkers(markerCorners, size_of_marker, cameraMatrix=mtx_camera, distCoeffs=distortion)

    #dessin des axes
    if(ecritur):
        outputImage = image.copy()
        length_of_axis = 0.02
        for i in range(len(tvecs)):
            cv.aruco.drawAxis(outputImage, mtx_camera, distortion, rvecs[i], tvecs[i], length_of_axis)
        #print("enregistrement de l'image avec axe terminée")
        cv.imwrite(nom_out_axe , outputImage)

    ###  debut projection 3D -> 2D  ###
    #trouvé ofset entre l'origine theorique bord gauche en haut et l'origine du code centre d'un aruco
    pos_aruc_x = id_origin % 5
    pos_aruc_y = int(id_origin / 5)
    x_offset = pos_aruc_x * -3.7 - 2.9/2
    y_offset = pos_aruc_y * 3.7 + 2.9/2
    offset = np.float32([x_offset, y_offset, 0]) ###  hauteur robot 1.2, entre aruco 37mm coté 29
    
    #mise en forme des position donné
    obj_bleu = np.float32(obj_bleu)
    obj_vert = np.float32(obj_vert)

    zero = np.float32([[0,0,0]]).reshape(-1,3)
    objpts2 = add_offset(obj_bleu, offset).reshape(-1,3) / 100
    objpts = add_offset(obj_vert, offset).reshape(-1,3) / 100

    #calcule des position sur l'image et dessin des point
    image_copy3 = image.copy()
    
    points2d_vert = dessin_pts_projete(objpts, image_copy3, rvecs, tvecs, mtx_camera, distortion, (0, 255, 0),ecritur = ecritur)
    points2d_bleu = dessin_pts_projete(objpts2, image_copy3, rvecs, tvecs, mtx_camera, distortion, (255, 255, 0),ecritur = ecritur)
    dessin_pts_projete(zero, image_copy3, rvecs, tvecs, mtx_camera, distortion, (0, 0, 255), ecritur = ecritur)

    #print( "fin de projection pour l'image ", nom_im)
    if(ecritur): cv.imwrite(nom_out_detect, image_copy3)

    return (points2d_bleu, points2d_vert)

def find_pos_3d_to_2d_seul_axe(nom_im = "image_exemple.png", nom_out_aruco = " ", nom_out_axe = " ", 
    nom_out_detect = " ", obj_bleu = [[2,5,0],[1,3,0]] , obj_vert = [[5,5,0],[0,0,0]], ecritur = False):
    
    #generer les nom
    if(ecritur):
        if(nom_out_aruco == " "):
            nom_out_aruco = "marker_detect_" + nom_im

        if(nom_out_axe == " "):
            nom_out_axe = "axis_detect_" + nom_im

        if(nom_out_detect == " "):
            nom_out_detect = "points3D_2_2D" + nom_im


    image = cv.imread(nom_im)

    # paramètre de la borde
    markersX = 5              # Number of markers in X direction
    markersY = 7              # Number of markers in Y direction
    markerLength = 0.029#60         # Marker side length (in meter)
    markerSeparation = 0.0075#15     # Separation between two consecutive markers in the grid (in meter)
    ar = cv.aruco.DICT_6X6_1000
    aruco_dict = cv.aruco.Dictionary_get(ar)  # Dictionary id
    axisLength = 0.5 * (min(markersX, markersY) * (markerLength + markerSeparation) + markerSeparation)

    #valeur de caméra
    mtx_camera = np.array([[886.773967, 0.000000, 627.460303], 
        [0.000000, 888.130640, 367.062129],
        [0.000000, 0.000000, 1.000000]])

    distortion = np.array([0.095401, -0.178426, -0.001307, -0.010675, 0.0])

    #creation de la bord
    board =  cv.aruco.GridBoard_create(markersX, markersY, markerLength, markerSeparation, aruco_dict)

    parameters =  cv.aruco.DetectorParameters_create()
    markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    #dessin des aruco
    if(ecritur):
        outputImage = image.copy()
        cv.aruco.drawDetectedMarkers(outputImage, markerCorners, markerIds)

        cv.imwrite(nom_out_aruco, outputImage)

    #detection axe de chaque aruco
    rvecs,tvecs, trash = cv.aruco.estimatePoseSingleMarkers(markerCorners, markerLength, cameraMatrix=mtx_camera, distCoeffs=distortion)
    #detection d'un axe de la borde
    s, rvec,tvec = cv.aruco.estimatePoseBoard(markerCorners, markerIds, board, mtx_camera, distortion, rvecs, tvecs)
    
    #dessin des axes
    if(ecritur):
        outputImage = image.copy()
        cv.aruco.drawAxis(outputImage, mtx_camera, distortion, rvec, tvec, axisLength)
        #print("enregistrement de l'image avec axe terminée")
        cv.imwrite(nom_out_axe , outputImage)

    
    ###  debut projection 3D -> 2D  ###
    #mise en forme des donné
    offset = np.float32([0, 25.1 , 0]) ###  hauteur robot 1.2, entre aruco 3.7mm coté 29mm
    obj_bleu = np.float32(obj_bleu)
    obj_vert = np.float32(obj_vert)

    zero = np.float32([[0,0,0]]).reshape(-1,3)
    objpts2 = add_offset(obj_bleu, offset).reshape(-1,3) / 100
    objpts = add_offset(obj_vert, offset).reshape(-1,3) / 100

    image_copy3 = image.copy()

    #calcule des position sur l'image et dessin des point
    points2d_vert = dessin_pts_projete(objpts, image_copy3, rvec, tvec, mtx_camera, distortion, (0, 255, 0),1, ecritur = ecritur)
    points2d_bleu = dessin_pts_projete(objpts2, image_copy3, rvec, tvec, mtx_camera, distortion, (255, 255, 0),1, ecritur = ecritur)

    #print( "fin de projection pour l'image ", nom_im)
    if(ecritur):cv.imwrite(nom_out_detect, image_copy3)

    return (points2d_bleu, points2d_vert)

def dessin_pts_projete(pts, image_copy3, rvecs, tvecs, mtx_camera, distortion, color, B = 0, ecritur = True):
    if(B == 0):
        #projection 
        points2d, _ = cv.projectPoints(pts, rvecs[-1], tvecs[-1], mtx_camera, distortion)
    else:
        #mise en forme du tableau (parentèse en trop)
        R = rvecs
        T = tvecs
        RR = np.zeros((1,len(R)))
        TT = np.zeros((1,len(R)))
        for i in range(len(R)):
            RR[0][i] = R[i][0]
            TT[0][i] = T[i][0]
        #projection
        points2d, _ = cv.projectPoints(pts, RR, TT, mtx_camera, distortion)
    
    #dessin des point trouvé
    if(ecritur):
        for j, point in enumerate(points2d): # loop over the points
        #draw a circle on the current contour coordinate
            cv.circle(image_copy3, (int(point[0][0]), int(point[0][1])), 2, color, 2, cv.LINE_AA)
    
    #pour enlever les parnetese en trop
    A = points2d.astype(int)
    AA = np.zeros((len(A),2))
    for i in range(len(A)):
        #AA.append(A[i][0])
        AA[i][0] = A[i][0][0]
        AA[i][1] = A[i][0][1]

    return AA

def trouve_pos_exact(xy, name_img = "image_exemple.png", name_out = " ", h = 25, w = 25, aire_min = 16,  aire_max = 30, ecritur = False):
    
    if(ecritur):
        if(name_out == " "):
            name_out = "avec_point_" + name_img

    img = cv.imread(name_img, cv.IMREAD_COLOR)
    #taille de l'image
    height, width = img.shape[:2]

    xy_reel_tot = []
    warning = 0
    impressision = 0

    for coord in xy:
        x = int(coord[0])
        y = int(coord[1])
        
        #trouve les bort pour decoupage au tour du point
        bas = y-h if(y-h > 0) else 0   
        haut = y+h if(y+h < height) else height

        gauche = x-w if(x-w > 0) else 0
        droite = x+w if(x+w < width) else width

        #decoupage de l'image
        cropped = img.copy()[bas:haut, gauche:droite]

        #cv.imwrite("cropped_" + str(x) + "_" + str(y) + ".png", cropped)

        #image en noir et blanc
        img_gray1 = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

        #trouve contour
        ret, thresh = cv.threshold(img_gray1, 75, 255, cv.THRESH_BINARY) #2eme paramètre a retravailler pour ajuster contraste
        #thresh = cv.adaptiveThreshold(img_gray1,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

        #cv.imwrite("aaa_" + str(x) + "_" + str(y) + "thresh.png", thresh)

        contours,h1 = cv.findContours(thresh,1,2)
        xy_reel = []
        for cnt in contours:
            #teste si le contour est valide (par le rayon du cercle circonscrit et l'aire)
            (a,b),radius = cv.minEnclosingCircle(cnt)
            A = cv.contourArea(cnt)
            if(A<aire_max and A>aire_min and radius<8 and radius>1) :
                approx = cv.approxPolyDP(cnt,0.01*cv.arcLength(cnt,True),True)
                #cv.drawContours(cropped,[cnt],0,255,-1)
                
                #calcule de centre gravité du coutour
                M = cv.moments(cnt)
                if(M['m00']>0) :
                    cx = int(M['m10']/M['m00']) + gauche
                    cy = int(M['m01']/M['m00']) + bas       #coordoné de l'image entiere   
                    xy_reel.append([cx, cy])
        #corrige le resultat si il y y plusieur contour valide ou aucun    
        if (len(xy_reel) == 1) :    #ok
            xy_reel_tot.append(xy_reel[0])
        else: 
            #xy_reel_tot.append([x, y])
            if (len(xy_reel) == 0) :    #aucun point trouvé (posibilité d'erreur d'axe)
                warning += 1
                xy_reel_tot. append([x, y]) #prendre point aproximé
            else :                          #plusieur points trouvé on prend le plus proche de la valeur aproximé
                impressision += 1
                dist_min = 1000
                for c in xy_reel :
                    dist = np.square(x-c[0]) + np.square(y-c[1])
                    if(dist<dist_min):
                        dist_min = dist
                        cx = c[0]
                        cy = c[1]
                xy_reel_tot.append([cx, cy])
        if(ecritur): cv.circle(img, (xy_reel_tot[-1][0], xy_reel_tot[-1][1]), 1, (0, 255, 0), 2) 
        #cv.imwrite("cercle_detect_" + str(x) + "_" + str(y) + ".png", cropped)    
    if(ecritur):cv.imwrite(name_out, img)
    W = True if(warning>=3) else False
    I = True if(impressision>=3 or warning) else False

    #print(name_img, "a été traiter, les coordonés éxacte des borts des robots sont:")
    #print(xy_reel)
    return(xy_reel_tot,W,I)  

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)

    else:
        buf = input_img.copy()

    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
   
def detect_roombot(name_img, coefc, coefb, nom_out = " ", ecritur = False) :

    if(nom_out == " " and ecritur):
        nom_out = "roombot_detect_" + name_img

    img = cv.imread(name_img)

    b = 0
    c = 0

    out = np.zeros((128*2, 128*3, 3), dtype = np.uint8)
    
    out = apply_brightness_contrast(img, b, c)
    mask_1 = cv.inRange(out, (0, 50, 0), (50, 255,50))
    mask_overlay = mask_overlayc = mask_overlayb = mask_1
        
    for i in range(coefc) :
        for j in range(coefb) :
            out = apply_brightness_contrast(img, b, c + i)
            maskc = cv.inRange(out, (0, 50, 0), (50, 255,50))
            mask_overlayc = cv.addWeighted(mask_overlayc,0.5,maskc,0.5,0)
            out = apply_brightness_contrast(img, b + j, c)
            maskb = cv.inRange(out, (0, 50, 0), (50, 255,50))
            mask_overlayb = cv.addWeighted(mask_overlayb,0.5,maskb,0.5,0)
        
    #mask_overlay = cv.addWeighted(mask_overlayb,0.5,mask_overlayc,0.5,0)
    mask_overlay = cv.add(mask_overlayb,mask_overlayc)
    
    if(ecritur): cv.imwrite(nom_out, mask_overlay)
    return(mask_overlay)

def trouve_pos_exact_roombot(name_img, xy, nom_out_contrast = " ", nom_out_contour = " ", name_out_pts = " ", ecritur = False):

    if(ecritur):
        if(nom_out_contrast == " "):
            nom_out_contrast = "contrast_" + name_img
        if(nom_out_contour == " "):
            nom_out_contour = "contour_" + name_img
        if(name_out_pts == " "):
            name_out_pts = "ok_" + name_img

    img = detect_roombot(name_img, coefc = 25, coefb = 5, nom_out = nom_out_contrast, ecritur=ecritur)
    if(ecritur): 
        cv.imwrite(nom_out_contrast, img)
        img = cv.imread(nom_out_contrast, cv.IMREAD_COLOR)
        img2 = cv.imread(name_img)
    else:
       cv.imwrite("contraste.png", img) 
       img = cv.imread("contraste.png", cv.IMREAD_COLOR)
    
    h = w = 50
    
    #taille de l'image
    height, width = img.shape[:2]

    xy_reel_tot = []
    warning = 0
    impressision = 0

    for coord in xy:
        x = int(coord[0])
        y = int(coord[1])

        #trouve les bort pour decoupage au tour du point
        bas = y-h if(y-h > 0) else 0   
        haut = y+h if(y+h < height) else height

        gauche = x-w if(x-w > 0) else 0
        droite = x+w if(x+w < width) else width

        #decoupage de l'image
        cropped = img[bas:haut, gauche:droite]
        cv.imwrite("cropped.png", cropped)

        img_gray1 = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

        #trouve contour
        ret, thresh = cv.threshold(img_gray1, 75, 255, cv.THRESH_BINARY) #2eme paramètre a retravailler pour ajuster contraste
        cv.imwrite("thresh.png", thresh)

        contours,h1 = cv.findContours(thresh,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        grand_contour = []
        centre_contour = []
        xy_reel = []
        for cnt in contours :
            A = cv.contourArea(cnt)
            #print(A)
            if(A>1500):     #grand contour
                grand_contour.append(cnt)

            if(A>100 and A<300):
                centre_contour.append(cnt)
                M = cv.moments(cnt)
                if(M['m00']>0) :
                    cx = int(M['m10']/M['m00']) + gauche
                    cy = int(M['m01']/M['m00']) + bas       #coordoné de l'image entiere   
                    xy_reel.append([cx, cy])
        
        if (len(xy_reel) == 1) :    #ok
            xy_reel_tot.append(xy_reel[0])
        else: 
            #xy_reel_tot.append([x, y])
            if (len(xy_reel) == 0) :    #aucun point trouvé (posibilité d'erreur d'axe)
                warning += 1
                xy_reel_tot. append([x, y]) #prendre point aproximé
            else :                          #plusieur points trouvé on prend le plus proche de la valeur aproximé
                impressision += 1
                dist_min = 1000
                for c in xy_reel :
                    dist = np.square(x-c[0]) + np.square(y-c[1])
                    if(dist<dist_min):
                        dist_min = dist
                        cx = c[0]
                        cy = c[1]
                xy_reel_tot.append([cx, cy])
        if(ecritur): 
            cv.drawContours(cropped, grand_contour , -1, (255, 0, 255), 2, cv.LINE_AA)
            cv.drawContours(cropped, centre_contour , -1, (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(img2, (xy_reel_tot[-1][0], xy_reel_tot[-1][1]), 1, (0, 255, 0), 2)
    if(ecritur):
        cv.imwrite(name_out_pts, img2)
        cv.imwrite(nom_out_contour, img)
    W = True if(warning>=3) else False
    I = True if(impressision>=3 or warning) else False

    return(xy_reel_tot,W,I)

def trouve_oriantation_roombot(xy, ecritur = False, name_img = ""):
    l = 20
    vec = []
    for i in range(3):
        x = xy[i][0]-xy[i+1][0]
        y = xy[i][1]-xy[i+1][1]
        abs = math.sqrt(x*x + y*y)
        vec.append([x/abs, y/abs])
    vec.append([x/abs, y/abs])
    
    if(ecritur):
        img = cv.imread(name_img)
        for i in range(4) :
            x1 = xy[i][0]
            y1 = xy[i][1]

            x2 = x1 + vec[i][0] * l
            y2 = y1 + vec[i][1] * l
            cv.arrowedLine(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2, cv.LINE_AA)
        cv.imwrite(name_img, img)
    return vec

def trouve_oriantation_roombot2(pos_centre, pos_axe, pos_centre_reel, name_img, name_out, ecritur = False):
    vec_x = compt_all_offset(pos_centre, pos_axe[:4])
    vec_y = compt_all_offset(pos_centre, pos_axe[4:])
    if(ecritur):

        pos_axe_reel_x = add_all_offset_2D(pos_centre_reel,vec_x)
        pos_axe_reel_y = add_all_offset_2D(pos_centre_reel,vec_y)
        img = cv.imread(name_img)
        for i in range(len(pos_centre_reel)):
            cv.arrowedLine(img, (int(pos_centre_reel[i][0]), int(pos_centre_reel[i][1])), 
                (int(pos_axe_reel_x[i][0]), int(pos_axe_reel_x[i][1])), (0, 0, 255), 2, cv.LINE_AA)
            cv.arrowedLine(img, (int(pos_centre_reel[i][0]), int(pos_centre_reel[i][1])), 
                (int(pos_axe_reel_y[i][0]), int(pos_axe_reel_y[i][1])), (0, 255, 0), 2, cv.LINE_AA)

        cv.imwrite(name_out, img)
    return(vec_x, vec_y)

def add_offset(A, v):

    for i in range(len(A)):
        A[i] = A[i] + v

    return(A)

def compt_offset(pt1, pt2):
    v = [pt2[0]-pt1[0],pt2[1]-pt1[1]]
    return(v)

def compt_all_offset(pts1, pts2):
    l = len(pts1)
    v = []
    #print(v)
    for i in range(l):
        coord_pts1 = pts1[i]
        coord_pts2 = pts2[i]
        x1 = int(coord_pts1[0])
        y1 = int(coord_pts1[1])
        x2 = int(coord_pts2[0])
        y2 = int(coord_pts2[1])
        v.append(compt_offset([x1,y1], [x2,y2]))
        
    return(v)

def add_offset_2D(pt, ofset):
    pt_new = []
    for i in range(len(pt)):
        pt_new.append(pt[i] + ofset[i])
    return(pt_new)

def add_all_offset_2D(pts, ofsets):
    l = len(pts)
    pts_new = []
    for i in range(l):
        pts_new.append(add_offset_2D(pts[i], ofsets[i]))
    return(pts_new)

def concat_all_element(lis_in):
    lis_out = []
    for li in lis_in:
        for a in li:
            lis_out.append(a)
    return lis_out

def dessine_poin_and_axes(data, name_img, nom_out):
    img = cv.imread(name_img)

    for ii in range(4):
        x = data[ii*2]
        y = data[1+ii*2]
        vecxx = data[8+ii*2]
        vecxy = data[9+ii*2]
        vecyx = data[16+ii*2]
        vecyy = data[17+ii*2]

        ptVecX = add_offset_2D([x,y], [vecxx, vecxy])
        ptVecY = add_offset_2D([x,y], [vecyx, vecyy])

        cv.circle(img, (x, y), 1, (0, 255, 0), 2)
        cv.arrowedLine(img, (int(x),int(y)), (int(ptVecX[0]),int(ptVecX[1])), (0, 0, 255), 2, cv.LINE_AA)
        cv.arrowedLine(img, (int(x),int(y)), (int(ptVecY[0]),int(ptVecY[1])), (0, 255, 0), 2, cv.LINE_AA)

    cv.imwrite(nom_out, img)