import modul_aruco
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # no UI backend
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math
import cv2 as cv
import os

def enleve_zeros(t,x):
    t1 = []
    x1 = []
    for i in range(len(x)):
        if(x[i] != 0):
            x1.append(x[i])
            t1.append(t[i])
    return(t1,x1)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def complette_data(data,code_erreur = 0):
    data1 = []
    i = 0
    while(i<len(data)):
        if(data[i] != code_erreur):
            i_valide1 = i
            data1.append(data[i])
        else:
            while(i<len(data)-1 and data[i] == code_erreur):
                i += 1
            i_valide2 = i
            nb_zeros = i_valide2 - i_valide1
            d1 = data[i_valide1]
            d2 = data[i_valide2]
            pas_data = (d2 - d1)/nb_zeros
            for ii in range (nb_zeros):
               data1.append(data[i_valide1] + pas_data*(ii+1))
            i_valide1 = i_valide2
        i += 1
    return data1

def dessine_data(data, fichier_in, fichier_out):
    print("TRAITEMENT DES IMAGES COMENCE")
    print("nom du fichier : " + fichier_out)
    #pour la barre de progretion
    nb_img = len(data)
    len_barre = 60-1
    step = nb_img / len_barre
    animation = ["-", "\\", "|", "/"]
    print("\x1b\x1b[0m")

    for i in range(nb_img):
        name_image = fichier_in +"/" + data.Name_of_image[i]
        nom_out = fichier_out +"/" + data.Name_of_image[i]

        ligne = []
        for col in data.columns[2:26]:
            ligne.append(data[col][i])

        modul_aruco.dessine_poin_and_axes(ligne, name_image, nom_out)

        anim = animation[i % 4] if i < nb_img else ""
        print ("\x1b[1A\x1b[33;1m[%s]\x1b[2G%s%s\x1b[%iG\x1b[32m(%i/%i) \x1b[31" 
            % ("."*len_barre, "#"*(int(i / step)), anim, (len_barre + 4), i, nb_img))

######################################################
file_out_img_complet    = "1_Tout_image_detect"
file_out_img_filtr      = "2_Tout_image_filtr"
file_out_img_smoo_3     = "3_Tout_image_smoo_3"
file_out_img_smoo_19    = "4_Tout_image_smoo_19"
file_out_img_colle      = "5_Tout_image_collee"

filename = 'Csv_data.csv'
flag_ecritur = False

data = pd.read_csv(filename)
data_compl = data.copy()
data_filtr = data.copy()
data_smoo_3 = data.copy()
data_smoo_19 = data.copy()

t = range(len(data))

for col in data.columns[2:10] :
    lis = data[col]
    lis_complet = complette_data(lis)

    #calcule d'approximation
    lis_filtr = savgol_filter(lis_complet, 51, 2)
    lis_smoo_3 = smooth(lis_complet,3)
    lis_smoo_19 = smooth(lis_complet,19)
    #arrondi et passage en int
    lis_complet = list(np.int_(np.round(lis_complet)))
    lis_filtr = list(np.int_(np.round(lis_filtr)))
    lis_smoo_3 = list(np.int_(np.round(lis_smoo_3)))
    lis_smoo_19 = list(np.int_(np.round(lis_smoo_19)))

    #enregitrement de donné dans une nouvel dataFrame
    data_compl[col] = lis_complet
    data_filtr[col] = lis_filtr
    data_smoo_3[col] = lis_smoo_3
    data_smoo_19[col] = lis_smoo_19

    if (flag_ecritur):
        plt.clf()
        plt.plot(t,lis_complet)
        plt.plot(t,lis_filtr)
        plt.title(col + ' filtre polinomiale')
        plt.savefig(col + "_filtre_polinomiale.png")  #savefig, don't show

        plt.clf()
        plt.plot(t,lis_complet)
        plt.plot(t,lis_smoo_3)
        plt.title(col + ' filtre smooth 3')
        plt.savefig(col + "_filtre_smooth_3.png")  #savefig, don't show

        plt.clf()
        plt.plot(t,lis_complet)
        plt.plot(t,lis_smoo_19)
        plt.title(col + ' filtre smooth 19')
        plt.savefig(col + "_filtre_smooth_19.png")  #savefig, don't show

for col in data.columns[10:] :
    lis = data[col]
    lis_complet = complette_data(lis,1000)

    #calcule d'approximation
    lis_filtr = savgol_filter(lis_complet, 51, 2)
    lis_smoo_3 = smooth(lis_complet,3)
    lis_smoo_19 = smooth(lis_complet,19)
    #arrondi et passage en int
    lis_complet = list(np.int_(np.round(lis_complet)))
    lis_filtr = list(np.int_(np.round(lis_filtr)))
    lis_smoo_3 = list(np.int_(np.round(lis_smoo_3)))
    lis_smoo_19 = list(np.int_(np.round(lis_smoo_19)))

    #enregitrement de donné dans une nouvel dataFrame
    data_compl[col] = lis_complet
    data_filtr[col] = lis_filtr
    data_smoo_3[col] = lis_smoo_3
    data_smoo_19[col] = lis_smoo_19

if os.path.exists('Csv_data_compl.csv'): os.remove("Csv_data_compl.csv")
if os.path.exists('Csv_data_filt.csv'): os.remove("Csv_data_filt.csv")
if os.path.exists('Csv_data_smoo_3.csv'): os.remove("Csv_data_smoo_3.csv")
if os.path.exists('Csv_data_smoo_19.csv'): os.remove("Csv_data_smoo_19.csv")

data_compl.to_csv("Csv_data_compl.csv", index=False)
data_filtr.to_csv("Csv_data_filt.csv", index=False)
data_smoo_3.to_csv("Csv_data_smoo_3.csv", index=False)
data_smoo_19.to_csv("Csv_data_smoo_19.csv", index=False)


dessine_data(data_compl, "image", file_out_img_complet)
dessine_data(data_filtr, "image", file_out_img_filtr)
dessine_data(data_smoo_3, "image", file_out_img_smoo_3)
dessine_data(data_smoo_19, "image", file_out_img_smoo_19)



print("COLLAGE DES IMAGES COMENCE")
print("nom du fichier : " + file_out_img_colle)
#pour la barre de progretion
nb_img = len(data)
len_barre = 60-1
step = nb_img / len_barre
animation = ["-", "\\", "|", "/"]
print("\x1b\x1b[0m")
i = 0

for nom_im in data.Name_of_image:

    img_comp = cv.imread(file_out_img_complet + "/" + nom_im)
    img_filt = cv.imread(file_out_img_filtr + "/" + nom_im)
    img_smoo_3 = cv.imread(file_out_img_smoo_3 + "/" + nom_im)
    img_smoo_19 = cv.imread(file_out_img_smoo_19 + "/" + nom_im)

    col1 = np.vstack([img_comp, img_filt])
    col2 = np.vstack([img_smoo_3, img_smoo_19])
    collage = np.hstack([col1, col2])

    cv.imwrite(file_out_img_colle + "/" + nom_im, collage)
    i += 1
    anim = animation[i % 4] if i < nb_img else ""
    print ("\x1b[1A\x1b[33;1m[%s]\x1b[2G%s%s\x1b[%iG\x1b[32m(%i/%i) \x1b[31" 
        % ("."*len_barre, "#"*(int(i / step)), anim, (len_barre + 4), i, nb_img))