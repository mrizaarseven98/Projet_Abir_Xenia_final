import os
name_repertori = ["image", "image_aruco_detect", "image_axe_detect", "image_contour", "image_pts_3Dto2D", "image_treter"]
for repertoire in name_repertori:
    if not os.path.exists(repertoire):
        os.makedirs(repertoire)
