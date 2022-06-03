#detecté les aruco sur chaque image d'un fichier ansie que les axe.
#Puis trouve d'abbord les coordoné aproximé des points clé des rebot, puis affine les meusures
import pandas as pd
import os
from telnetlib import NOP
from datetime import datetime
import modul_aruco
import numpy as np
import time

#paramètre des emplacement à changer selon préférance
file_imput = "image"
file_out_rombot_detect   = "image_roombot_detect"
file_out_aruco_detect   = "image_aruco_detect"
file_out_axe_detect     = "image_axe_detect"
file_out_pts_3Dto2D     = "image_pts_3Dto2D"
file_out_img_ok         = "image_treter"
file_out_contour        = "image_contour"

name_txt_file = "data_roombot.txt"
filename = 'Csv_data.csv'
flag_ecritur = False

tps1 = time.process_time()

#les coordoné 3D des points a placer en cm
h = 11        #hauteur du robeaut
l = 2.1       #largeur de la marge
d = 3.25       #diametre du cercle vert
pts_centre = [[0-l,22,h], [0-l,11,h], [11-l,11,h], [22-l,11,h]]
pts_axe = [[0-l+d,22,h], [0-l+d,11,h], [11-l+d,11,h], [22-l+d,11,h], [0-l,22+d,h], [0-l,11+d,h], [11-l,11+d,h], [22-l,11+d,h]]

#gestion fichier
fichier = open(name_txt_file, "w")#fichier dans le qulle on note les valuers

fichier.write("Donné image triangle")
fichier.write("\n\nAuteur: \t code_complet_seul_axe.py")
fichier.write("\nDate: \t\t "+ str(datetime.now()))
fichier.write("\nEnregitrement d'image: \t "+ ("oui"if(flag_ecritur) else"non"))
fichier.write("\n\nLEGENDE: \nV: \t image trêtée avec sucsess")
fichier.write("\nI: \t image trêtée avec sucsess mais il y a eu des point parasite détecter donc il y a possibilité d'impresision")
fichier.write("\nW: \t il y a surment eu des problème de traitement\n")
fichier.write("\nF: \t Erreur sur le traitement d'image\n")


#.scv

header = ['Name_of_image', 'statua', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4',
         'vxx1', 'vxy1', 'vxx2', 'vxy2', 'vxx3', 'vxy3', 'vxx4', 'vxy4',
         'vyx1', 'vyy1', 'vyx2', 'vyy2', 'vyx3', 'vyy3', 'vyx4', 'vyy4']
data = []

tout_image = os.listdir(file_imput) #liste de tout les nom d'images

#pour barre de progression
nb_img = len(tout_image)
i = 0
len_barre = 60-1
step = nb_img / len_barre
num_errors = 0
num_warning = 0
animation = ["-", "\\", "|", "/"]
print("\x1b\x1b[0m")


for img in tout_image:

    nom_img_input = file_imput +"/" + img
    
    nom_out_roombot = file_out_rombot_detect + "/" + "contrast_" + img
    nom_out_aruco = file_out_aruco_detect + "/" + "aruco_" + img
    nom_out_axe = file_out_axe_detect + "/" + "axe_" + img
    nom_out_detect = file_out_pts_3Dto2D + "/" + "detect_" + img
    nom_out_contour = file_out_contour + "/" + "contour_" + img
    nom_out_img_ok = file_out_img_ok + "/" + "ok_" + img
    nom_out_img_ok2 = file_out_img_ok + "/" + "ok2_" + img
    try:
        pos_centre, pos_axe = modul_aruco.find_pos_3d_to_2d_seul_axe(nom_im = nom_img_input, 
            nom_out_aruco = nom_out_aruco, nom_out_axe = nom_out_axe, nom_out_detect = nom_out_detect, 
            obj_bleu = pts_centre , obj_vert = pts_axe, ecritur=flag_ecritur)
        
        #mask_overlay = modul_aruco.detect_roombot(nom_img_input, coefc = 25, coefb = 5, nom_out = nom_out_roombot, ecritur=flag_ecritur)
        pos_centre_reel,W,I = modul_aruco.trouve_pos_exact_roombot(nom_img_input, pos_centre, nom_out_contrast = nom_out_roombot, 
            nom_out_contour = nom_out_contour, name_out_pts = nom_out_img_ok,ecritur = flag_ecritur)

        nom_img = nom_out_img_ok if flag_ecritur==True else nom_img_input
        vec_x, vec_y = modul_aruco.trouve_oriantation_roombot2(pos_centre, pos_axe, pos_centre_reel, nom_img, nom_out_img_ok, ecritur = True)

        #pos_cotee_reel,W1,I1 = modul_aruco.trouve_pos_exact_roombot(nom_img_input, pos_origine, nom_out_contrast = nom_out_roombot, 
        #    nom_out_contour = nom_out_contour, name_out_pts = nom_out_img_ok,ecritur = flag_ecritur)
        
        #pos_coint_reel,W,I = modul_aruco.trouve_pos_exact(pos_coin, name_img = nom_img_input, name_out = nom_out_img_ok, h = 30, w = 30, aire_min = 30,  aire_max = 70, ecritur=True)
        if(W):
            status = "W"
            num_warning += 1
        else:
            if(I):
                status = "I" 
            else:
                status = "V"  
        fichier.write("\n" + status + "\tId = " + str(0) + "\t" + img + " : \t\t Centre:\t" +  str(pos_centre_reel) + "\t\t\t axe X: \t" +  str(vec_x) + "\t\t\t axe Y: \t" +  str(vec_y))
        data.append([img,status] + modul_aruco.concat_all_element(pos_centre_reel + vec_x + vec_y))

        #affichage barre de progression
        i+=1
        x = int(len_barre*i/nb_img)
        err_text = "err: " + str(num_errors)
        war_text = "war: " + str(num_warning)
        anim = animation[i % 4] if i < nb_img else ""
        print ("\x1b[1A\x1b[33;1m[%s]\x1b[2G%s%s\x1b[%iG\x1b[32m(%i/%i) \x1b[31;1m%s\x1b[0m\x1b[31;1m%s\x1b[36m" 
            % ("."*len_barre, "#"*(int(i / step)), anim, (len_barre + 4), i, nb_img, err_text, "\t " + war_text))
    
    except Exception as err:
        i+=1
        num_errors = num_errors + 1
        print("\x1b[1A\x1b[31;1mimage %s: %s\x1b[0J\x1b[1B\x1b[0m" % (img, "n'a pas pu être trété"))
        fichier.write("\nF\t\t\t\t\t" + img + " : \t" +  "pas trouvé")
        data.append([img,status] + [0,0,0,0,0,0,0,0, 1000,1000,1000,1000,1000,1000,1000,1000, 1000,1000,1000,1000,1000,1000,1000,1000])
    #modul_aruco.trouve_oriantation_roombot(pos_centre_reel, ecritur = True, name_img = nom_out_img_ok)
    


data = pd.DataFrame(data, columns=header)
data.to_csv(filename, index=False)




juste = (nb_img - num_errors - num_warning)*100/nb_img 
text_fin = "poursantage de réucite = " + str(juste) + "%"
print("PROCESS COMPLETE")
print(text_fin)
fichier.write("\n\n\n" + text_fin)

tps2 = time.process_time()

print("temps d'execution : ", tps2 - tps1, "s")
fichier.write("\ntemps d'execution : "+ str(tps2 - tps1) + " s")
fichier.close()
