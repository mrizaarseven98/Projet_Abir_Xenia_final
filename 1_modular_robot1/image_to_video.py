import cv2
import numpy as np
import os
from os.path import isfile, join

pathIn= 'image_treter/'
pathOut = 'video_modular_robot.mp4'
fps = 24


frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
#for sorting the file names properly
files.sort(key = lambda x: x[5:-4])
files.sort()
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
#for sorting the file names properly
#height, width, layers = img.shape
width = 1280
height = 720
size = (width,height)
files.sort(key = lambda x: x[5:-4])
if os.path.exists(pathOut): os.remove(pathOut)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

print("ENREGISTREMENT DE VIDEO")
print("nom du fichier : " + pathOut)
#pour la barre de progretion
nb_img = int(len(files))
len_barre = 60-1
step = nb_img / len_barre
animation = ["-", "\\", "|", "/"]
print("\x1b\x1b[0m")
i = 0

for i in range(nb_img):
    filename=pathIn + files[i]
    #reading each files
    img = cv2.imread(filename)
    #print("image ", filename,"a été enregistré")
    #inserting the frames into an image array
    frame_array.append(img)
    # writing to a image array
    #print(i, " image importé")

    out.write(img)
    anim = animation[i % 4] if i < nb_img else ""
    print ("\x1b[1A\x1b[33;1m[%s]\x1b[2G%s%s\x1b[%iG\x1b[32m(%i/%i) \x1b[31" 
        % ("."*len_barre, "#"*(int(i / step)), anim, (len_barre + 4), i, nb_img))
out.release()