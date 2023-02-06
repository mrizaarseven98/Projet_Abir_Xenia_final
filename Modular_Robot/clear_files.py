import shutil
import os
name_repertori = ["image_aruco_detect", "image_axe_detect", "image_contour", "image_thresh", "image_pts_3Dto2D", "image_treter","__pycache__"]
for repertoire in name_repertori:
    if  os.path.exists(repertoire):
        shutil.rmtree(repertoire)
