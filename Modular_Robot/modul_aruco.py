import cv2 as cv
import numpy as np
import subprocess
import yaml
import rosbag
from cv_bridge import CvBridge
from skspatial.measurement import area_signed

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

def calibration_aruco_size(nom_im = "image/frame0736.jpg"):

    img=cv.imread(nom_im)
    img_hsv= cv.cvtColor(img,cv.COLOR_BGR2HSV)
    

    total_brightness=np.average(img_hsv[:,:,2])
    threshold_constant=-90/np.exp(np.power(125/total_brightness,2.5)-1)
    kernel_size=int((85/1920)*img.shape[1])
    if kernel_size%2==0:
        kernel_size+=1

    markerCorners, markerIds=detect_aruco(nom_im)
    
    marker_area=1699.5/2073600*img.shape[0]*img.shape[1]
    realtime_marker_area=area_signed(markerCorners[-1][0])

    h_calib_number=30/np.sqrt(marker_area)*np.sqrt(realtime_marker_area)
    w_calib_number=30/np.sqrt(marker_area)*np.sqrt(realtime_marker_area)
    max_area_calib_numb=140/marker_area*realtime_marker_area
    min_area_calib_number=25/marker_area*realtime_marker_area
    max_radius_calib_number=8/np.sqrt(marker_area)*np.sqrt(realtime_marker_area)
    min_radius_calib_number=1/np.sqrt(marker_area)*np.sqrt(realtime_marker_area)

    return list([h_calib_number, w_calib_number, max_area_calib_numb, min_area_calib_number,
     max_radius_calib_number, min_radius_calib_number, kernel_size, threshold_constant])

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
    mtx_camera = np.array([[1758.7121537923381, 0.000000, 901.58508432724011], 
        [0.000000, 1758.7121537923381, 554.25871381481932],
        [0.000000, 0.000000, 1.000000]])

    distortion = np.array([-0.098834022104455549, -0.24592544849861744, 0.0034931988061491677, -0.0059880442722622584, 1.3957362410111847])

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
    mtx_camera = np.array([[1758.7121537923381, 0.000000, 901.58508432724011], 
        [0.000000, 1758.7121537923381, 554.25871381481932],
        [0.000000, 0.000000, 1.000000]])

    distortion = np.array([-0.098834022104455549, -0.24592544849861744, 0.0034931988061491677, -0.0059880442722622584, 1.3957362410111847])

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

def add_offset(A, v):

    for i in range(len(A)):
        A[i] = A[i] + v
    
    return(A)

def trouve_pos_exact(xy, name_img = "image_exemple.png", name_out = " ", h = 25, w = 25, aire_min = 16,  aire_max = 30, ecritur = False):
    
    if(ecritur):
        if(name_out == " "):
            name_out = "avec_point_" + name_img

    img = cv.imread(name_img, cv.IMREAD_COLOR)
    #imgb = cv.GaussianBlur(img,(3,3),0)
    """ img_gray1 = cv.cvtColor(imgb, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(img_gray1,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,15,-16) """
    #taille de l'image
    height, width = img.shape[:2]
    

    xy_reel_tot = []
    warning = 0
    impressision = 0

    list_calib=calibration_aruco_size(name_img)

    for coord in xy:
        x = int(coord[0])
        y = int(coord[1])
        
        signal=True
        #trouve les bort pour decoupage au tour du point
        if(y < 0 or y > height or x < 0 or x> width):
            xy_reel_tot. append([None, None])
            signal=False
            #If the projected point is outside the image, write [0,0] to later delete it in Training_Dataset_Generator.py
            #and to pass the other steps
        
        else:
    
            bas=y-h if(y-h > 0) else 0
            haut = y+h if(y+h < height) else height
            gauche = x-w if(x-w > 0) else 0
            droite = x+w if(x+w < width) else width

            if(x-w < 0 and x+w> width):
                print("1eksi "+str(x-w))
                print("1arti "+str(x+w))

            "Bu weird croppingi kaldirmak icin x'in ve y nin eksi oldugu durumlari kaldir"

            #decoupage de l'image
            cropped = img.copy()[bas:haut, gauche:droite]

            #cv.imwrite("cropped_" + str(x) + "_" + str(y) + ".png", cropped)

            #image en noir et blanc
            img_gray1 = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

            #trouve contour
            #ret, thresh = cv.threshold(img_gray1, 75, 255, cv.THRESH_BINARY) #2eme paramètre a retravailler pour ajuster contraste
            thresh = cv.adaptiveThreshold(img_gray1,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,list_calib[6],list_calib[7])

            #cv.imwrite("aaa_" + str(x) + "_" + str(y) + "thresh.png", thresh)

            contours,h1 = cv.findContours(thresh,1,2)
            xy_reel = []
            for cnt in contours:
                #teste si le contour est valide (par le rayon du cercle circonscrit et l'aire)
                (a,b),radius = cv.minEnclosingCircle(cnt)
                A = cv.contourArea(cnt)
                if(A<aire_max and A>aire_min and radius<int(list_calib[4]) and radius>int(list_calib[5])) :
                    approx = cv.approxPolyDP(cnt,0.01*cv.arcLength(cnt,True),True)
                    cv.drawContours(cropped,[cnt],0,255,-1)
                    #cv.imwrite("image_contour/cnt_" + str(name_img) + "_" + str(droite) + "_aire_" + str(A) + "_aire_min_"+
                    #str(aire_min)+"_aire_max_"+str(aire_max)+"_rad_min_"+str(int(list_calib[5]))+"_rad_max_"+str(int(list_calib[4]))+ "_radius_" 			#+str(radius) + ".png", cropped)
                    
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
        if signal:
            if(ecritur): cv.circle(img, (xy_reel_tot[-1][0], xy_reel_tot[-1][1]), 1, (0, 255, 0), 2) 
        #cv.imwrite("cercle_detect_" + str(x) + "_" + str(y) + ".png", cropped)    
    if(ecritur):cv.imwrite(name_out, img)
    W = True if(warning>=3) else False
    I = True if(impressision>=3 or warning) else False

    #print(name_img, "a été traiter, les coordonés éxacte des borts des robots sont:")
    #print(xy_reel)
    
    return(xy_reel_tot,W,I)  
    
def drawPoseAxis(image, output_file_name, robot_corner_coordinates_2d, robot_corner_coordinates_3d= [[-4.4,1.5,1.2], [0.25, 9.7, 1.2], [5, 1.5, 1.2]], number_of_robots=3):
    
    image=cv.imread(image)

    axisPoints=[[0,0,0],[5,0,0],[0,5,0],[0,0,5]]

    robot_corner_coordinates_2d=np.asarray(robot_corner_coordinates_2d)

    robot_corner_coordinates_3d=np.float32(robot_corner_coordinates_3d)
    axisPoints = np.float32(axisPoints)
    robot_corner_coordinates_2d=np.float32(robot_corner_coordinates_2d)

    mtx_camera = np.array([[886.773967, 0.000000, 627.460303], 
        [0.000000, 888.130640, 367.062129],
        [0.000000, 0.000000, 1.000000]])

    distortion = np.array([0.095401, -0.178426, -0.001307, -0.010675, 0.0])

    n={"rvec":[],"tvec":[]}
    rgb_list=[(0,0,255),(0,255,0),(255,0,0)]
    
    for i in range(0, number_of_robots*3, 3):
        try: #passes to the other robot if one keypoint is not visible
            for x in robot_corner_coordinates_2d[i:i+3]:
                if(np.isnan(x[0]) or np.isnan(x[1])):
                    raise Exception()
        except Exception:
            continue

        #calculates the rotation & translation vector for the 3d to 2d projection of the keypoints
        ss, rvec, tvec= cv.solveP3P(robot_corner_coordinates_3d, robot_corner_coordinates_2d[i:i+3], mtx_camera, distortion,flags=5)
        
        #projects the axis from an arbitrary reference frame onto the robots
        axisPoints_2d, _ = cv.projectPoints(axisPoints, rvec[0], tvec[0], mtx_camera, distortion)

        n["rvec"].append(rvec)
        n["tvec"].append(tvec)

        for i in range(1,4):
            #draws the axis on the robots
            cv.arrowedLine(image, tuple(int(x) for x in axisPoints_2d[0][0]), tuple(int(y) for y in axisPoints_2d[i][0]), rgb_list[i-1],3)

        cv.imwrite(output_file_name,image)
    return n #returns the dictionary containing rvec and tvec
    

def crop_aruco(image, output_file_name, robot_2d_points):
    #image="image/frame0220.jpg"
    image_read=cv.imread(image)
    markerCorners, markerIds=detect_aruco(image)
    output_x=image_read.shape[1]
    output_y=image_read.shape[0]
   
    

    #print(str(np.min(markerCorners))+"wowowowowow \n \n")
    for i in range(0, np.shape(markerCorners)[0]):

        
        #print(str(output_x)+ "\n")
        if markerCorners[i][0][0][0] < output_x:
            output_x=markerCorners[i][0][0][0]

        if markerCorners[i][0][0][1] < output_y:
            output_y=markerCorners[i][0][0][1]

    """ print(str(np.min(markerCorners[:][0][0][0]))+"wowowowowow \n \n")
    print(str(output_x)+ "\n \n")
    print(str(robot_2d_points)+"robot"+"\n \n")
    print(str(np.max([robot_2d_points[i][1] for i in range(0,len(robot_2d_points))]))+"\n \n") """
    robot_2d_points_nparray = np.array(robot_2d_points, dtype=np.float64)
    """ print(str(np.nanmax([robot_2d_points_nparray[i][1] for i in range(0,len(robot_2d_points_nparray))]))+"\n wwwwowowo \n")
    print(str(output_x)+"\n") """
    if output_x>np.nanmax([robot_2d_points_nparray[i][0] for i in range(0,len(robot_2d_points_nparray))]):
        image_read1=image_read[:, :int(output_x)-50]

    elif output_y>np.nanmax([robot_2d_points_nparray[i][1] for i in range(0,len(robot_2d_points_nparray))]):
        image_read1=image_read[:int(output_y)-50, :]
    else:   
        raise Exception("cannot crop the aruco marker")



   
    #image_read1 = cv.resize(image_read1, (int(image_read1.shape[0]/3), int(image_read1.shape[1]/3)))
    cv.imwrite(output_file_name, image_read1)
    #cv.waitKey(0)
    
    