#detecté les aruco sur chaque image d'un fichier ansie que les axe.
#Puis trouve d'abbord les coordoné aproximé des points clé des rebot, puis affine les meusures

import os
from telnetlib import NOP
from datetime import datetime
import modul_aruco
import numpy as np
import time

#paramètre des emplacement à changer selon préférance
file_imput = "image"
file_out_aruco_detect   = "image_aruco_detect"
file_out_axe_detect     = "image_axe_detect"
file_out_pts_3Dto2D     = "image_pts_3Dto2D"
file_out_img_ok         = "image_treter"

name_txt_file = "data_triang_bord.txt"
flag_ecritur = True

tps1 = time.process_time()

#les coordoné 3D des points a placer en cm
h = 1.2         #hauteur du robeaut
pts_centre = [[0,10,h], [-15,25,h], [-30,0,h]]
#pts_coin = [[-4.4,11.5,h], [5,11.5,h], [0.25,19.7,h], [-15.25,15.3,h], [-20,23.5,h], [-10.6,23.5,h], [-28.5,4.4,h], [-28.5,-5,h], [-20.3,-0.25,h]]
pts_coin = [[-5,16.5,h], [-14.4,16.5,h], [-9.75,24.7,h], [-32.5,14.6,h], [-32.5,5.2,h], [-24.3,9.95,h], [-29.7,0.25,h], [-21.5,5,h], [-21.5,-4.4,h], [-32.5,-4.1,h], [-32.5,-13.5,h], [-24.3,-8.75,h]]

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

    nom_out_aruco = file_out_aruco_detect + "/" + "aruco_" + img
    nom_out_axe = file_out_axe_detect + "/" + "axe_" + img
    nom_out_detect = file_out_pts_3Dto2D + "/" + "detect_" + img
    nom_out_img_ok = file_out_img_ok + "/" + "ok_" + img

    try:
        pos_centre, pos_coin = modul_aruco.find_pos_3d_to_2d_seul_axe(nom_im = file_imput +"/" + img, 
        nom_out_aruco = nom_out_aruco, nom_out_axe = nom_out_axe, nom_out_detect = nom_out_detect, obj_bleu = pts_centre , obj_vert = pts_coin, ecritur=flag_ecritur)
    
        pos_coint_reel,W,I = modul_aruco.trouve_pos_exact(pos_coin, name_img = file_imput +"/" + img, name_out = nom_out_img_ok, h = 25, w = 25, aire_min = 70,  aire_max = 250, ecritur=True)
        if(W):
            status = "W"
            num_warning += 1
        else:
            if(I):
                status = "I" 
            else:
                status = "V"  
        fichier.write("\n" + status + "\tId = " + str(0) + "\t\t" + img + " : \t" +  str(pos_coint_reel))

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


juste = (nb_img - num_errors - num_warning)*100/nb_img 
text_fin = "poursantage de réucite = " + str(juste) + "%"
print("PROCESS COMPLETE")
print(text_fin)
fichier.write("\n\n\n" + text_fin)

tps2 = time.process_time()

print("temps d'execution : ", tps2 - tps1, "s")
fichier.write("\ntemps d'execution : "+ str(tps2 - tps1) + " s")
fichier.close()
