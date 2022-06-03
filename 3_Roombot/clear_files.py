import shutil
import os
name_repertori = ["1_Tout_image_detect", "2_Tout_image_filtr", "3_Tout_image_smoo_3", "4_Tout_image_smoo_19", "5_Tout_image_collee", "image", "image_aruco_detect", "image_axe_detect", "image_contour", "image_pts_3Dto2D", "image_roombot_detect", "image_treter"]
for repertoire in name_repertori:
    if  os.path.exists(repertoire):
        shutil.rmtree(repertoire)
