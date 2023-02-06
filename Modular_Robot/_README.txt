
#pour generer les image et les tréter
#Execute in the following order

python3 make_files.py
python3 video_to_image.py
python3 code_complet_seul_axe.py
python3 image_to_video.py
python3 Training_Dataset_Generator.py F F
#Delete the unwanted frames in /image_accurate_to_correct
python3 Training_Dataset_Generator.py T F
python3 KMeans1.py
python3 Training_Dataset_Generator.py T T 

#pour effacer tout les repertoir
python3 clear_files.py
