import imaplib
from sklearn.cluster import KMeans
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import random
import shutil
from tqdm import tqdm
import time

start = time.time()

path_training="training_images"
files_images_training=os.listdir(path_training)

path_training_downsized="training_images_downsized"



if os.path.exists(path_training_downsized):
    shutil.rmtree(path_training_downsized)

os.mkdir(path_training_downsized)


for file in tqdm(files_images_training, desc='Downscaling Training Images: '):

    img=cv.imread(path_training+"/"+file, cv.IMREAD_UNCHANGED)
    scale_percent = 30 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized=cv.resize(img, dim, interpolation = cv.INTER_AREA)
    cv.imwrite(path_training_downsized+"/"+file, resized)

dir_name_0="clustered_training_images"
if os.path.exists(dir_name_0):
    shutil.rmtree(dir_name_0)

os.mkdir(dir_name_0)


images=os.listdir(path_training_downsized)
img_size=cv.imread(path_training_downsized+"/"+images[0])
flattened=np.empty((len(images), img_size.shape[0]*img_size.shape[1]))

for i in tqdm(range(len(images)), desc="Reading Images"):
    img=cv.imread(path_training_downsized+"/"+images[i], cv.IMREAD_GRAYSCALE)
    flattened[i]=img.flatten(order='C')


print("KMeans clustering: This may take > 10min")

clusters=1200
kmeans = KMeans(n_clusters=clusters,init='random')
kmeans.fit(flattened)
Z = kmeans.predict(flattened)

#images=os.listdir("image")
print("Writing the files")
for i in range(0,clusters):
    
    row = np.where(Z==i)[0]  # row in Z for elements of cluster i
    random.shuffle(row)
    num = row.shape[0]       #  number of elements for each cluster
    t=0
    for k in range(0,num):
        if t<1:
            image_to_write=images[row[k]]
            shutil.copyfile("image/" + image_to_write, dir_name_0 + "/" +image_to_write)
        t+=1

end = time.time()
print(str(end - start)+" seconds")