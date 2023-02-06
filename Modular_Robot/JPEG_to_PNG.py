from PIL import Image
import os
import shutil
from tqdm import tqdm

dir1="reduced_training_images"
dir2=dir1+"_png"

if os.path.exists(dir2):
    shutil.rmtree(dir2)
os.mkdir(dir2)

images=os.listdir(dir1)

for img in tqdm(images, desc='Changing JPEG to PNG: '):
    im1 = Image.open(dir1+"/"+img)
    im1.save(dir2+"/"+img[:-4]+".png")
