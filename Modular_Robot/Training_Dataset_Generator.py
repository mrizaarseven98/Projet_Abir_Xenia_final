import numpy as np
import pandas as pd
import os, shutil
import cv2 as cv
from tqdm import tqdm
import sys



condition=bool("T"==sys.argv[1])
condition_cluster=bool("T"==sys.argv[2])


data=pd.read_csv('data_triang_bord.txt', skiprows=13, sep='\t', header=None,error_bad_lines=False)

v_data = data[data[0] == 'V']
v_data=v_data.dropna(how='all', axis=1)
v_data = v_data.drop(v_data.iloc[:, 0:2],axis = 1)
v_data.columns = [ "bodyparts", "Keypoints"]


#change this according to the number of robots you have
number_of_robots=1
#change this according to the possible number of robots in your training data
total_possible_robots= 4

def individual_labeler(number_of_robots):
    column_names=''
    for i in range(1,number_of_robots+1):
        column_names+="Robot"+str(i)+ '_left'+'_x'+", "
        column_names+="Robot"+str(i)+ '_left'+'_y'+", "
        column_names+="Robot"+str(i)+ '_right'+'_x'+", "
        column_names+="Robot"+str(i)+ '_right'+'_y'+", "
        column_names+="Robot"+str(i)+ '_front'+'_x'+", "
        if i==number_of_robots:
            column_names+="Robot"+str(i)+ '_front'+'_y'+""
        else:
            column_names+="Robot"+str(i)+ '_front'+'_y'+", "
            
    li = list(column_names.split(", "))
    return li

li=individual_labeler(number_of_robots)
v_data[li]=v_data.Keypoints.str.split(', ',expand=True)
v_data = v_data.drop(v_data.iloc[:, 1:2],axis = 1)

#addition 4th robot
#This block is a duplicate of a part of the above function. -> Bad code
#But couldn't find a workaround
if number_of_robots !=total_possible_robots:
  column_names=''
  for i in range(1,total_possible_robots-number_of_robots+1):
      column_names+="Robot"+ '_left'+'_1'+str(i)+", "
      column_names+="Robot"+ '_left'+'_2'+str(i)+", "
      column_names+="Robot"+ '_right'+'_3'+str(i)+", "
      column_names+="Robot"+ '_right'+'_4'+str(i)+", "
      column_names+="Robot"+ '_front'+'_5'+str(i)+", "
      if i==total_possible_robots-number_of_robots:
          column_names+="Robot"+ '_front'+'_6'+str(i)+""
      else:
          column_names+="Robot"+ '_front'+'_7'+str(i)+", "
          
  li1 = list(column_names.split(", "))
  for i in li1:
      v_data.insert(len(v_data.columns),i,"")

#deletes the characters "[" and "]" from keypoint data 
for i in range(len(li)):
    v_data[li[i]]=v_data[li[i]].str.replace("[][]","")

#deletes the ":" after .png
v_data["bodyparts"]=v_data["bodyparts"].str.replace(" : ","")


delimiter_robots=(6*number_of_robots)+1
v_data.columns.values[1:delimiter_robots]=v_data.columns[1:delimiter_robots].str[:-2]
v_data.columns.values[delimiter_robots:]=v_data.columns[delimiter_robots:].str[:-3]

#For same robots we would want to take the bodyparts of the robots as being the same
#This will allow to remove robot numbers from the table
v_data.columns.values[1:]=v_data.columns[1:].str.replace("[0123456789]","") #modify the list in the [] as 1,2,...,n_robots 
v_data=v_data.replace("None","")


""" v_data_temp=v_data.copy()
temp1=v_data_temp.iloc[0:,3]
v_data.iloc[0:,3]=v_data.iloc[0:,5]
v_data.iloc[0:,5]=temp1
temp2=v_data_temp.iloc[0:,4]
v_data.iloc[0:,4]=v_data.iloc[0:,6]
v_data.iloc[0:,6]=temp2

v_data_temp=v_data.copy()
temp3=v_data_temp.iloc[0:,9]
v_data.iloc[0:,9]=v_data.iloc[0:,11]
v_data.iloc[0:,11]=temp3
temp4=v_data_temp.iloc[0:,10]
v_data.iloc[0:,10]=v_data.iloc[0:,12]
v_data.iloc[0:,12]=temp4

v_data_temp=v_data.copy()
temp5=v_data_temp.iloc[0:,15]
v_data.iloc[0:,15]=v_data.iloc[0:,17]
v_data.iloc[0:,17]=temp5
temp6=v_data_temp.iloc[0:,16]
v_data.iloc[0:,16]=v_data.iloc[0:,18]
v_data.iloc[0:,18]=temp6

v_data_temp=v_data.copy()
temp7=v_data_temp.iloc[0:,21]
v_data.iloc[0:,21]=v_data.iloc[0:,23]
v_data.iloc[0:,23]=temp7
temp8=v_data_temp.iloc[0:,22]
v_data.iloc[0:,22]=v_data.iloc[0:,24]
v_data.iloc[0:,24]=temp8
 """


if not os.path.exists("image_accurate"):
    os.mkdir("image_accurate")

if condition==False:
    if os.path.exists("image_accurate_to_correct"):
      shutil.rmtree("image_accurate_to_correct")
    os.mkdir("image_accurate_to_correct")

files = v_data["bodyparts"]

for file in files:

    shutil.copy2('image_treter/ok_'+file, 'image_accurate')  #copy with metadata
    #print('copying.. ok_'+file+" to " + "image_accurate")
    if condition==False:
      shutil.copy2('image_treter/ok_'+file, 'image_accurate_to_correct')  #copy with metadata
    #print('copying.. ok_'+file+" to " + "image_accurate")

#Data transformation for DeepLabCut

l2=["Riza"] * (len(v_data.columns)-1)
l2.insert(0,"scorer")
v_data.loc[-3] = l2  # adding a row
v_data.index = v_data.index + 4  # shifting index
v_data.sort_index(inplace=True)
headers=v_data.columns.values.tolist()
v_data.columns=l2
v_data.iloc[0]=headers
v_data_n=v_data.copy()

cordlist=list()

for i in range((total_possible_robots*3)+1):
    if i==0:
        cordlist.append("coords")
    else:
        cordlist.append("x")
        cordlist.append("y")     


v_data_n.loc[3] = cordlist
v_data_n.sort_index(inplace=True)


individuals_list=individual_labeler(4)
individuals_list.insert(0,"individuals")
v_data_n.loc[0] = individuals_list
v_data_n.sort_index(inplace=True)
v_data_n.iloc[0,1:]=v_data_n.iloc[0,1:].str[0:6]

v_data_n.insert(1,"file_path1","")
v_data_n.insert(2,"file_path2","")
v_data_n

def multi_animal_transform(v_data_single):
    filepath1="video5"
    filepath2=v_data_single.iloc[3:,0]
    v_data_ma=v_data_single.copy()
    v_data_ma.iloc[3:,0]=["labeled-data"]*(len(v_data_ma)-3)
    v_data_ma.iloc[3:,1]=filepath1
    v_data_ma.iloc[3:,2]=filepath2
    v_data_ma.columns.values[1:3]=v_data_ma.columns[1:2].str[:-10]

    return v_data_ma

v_data_n=multi_animal_transform(v_data_n)
#v_data_reduced=multi_animal_transform(v_data_reduced)
v_data_n_toWrite=v_data_n.copy()

if not condition_cluster:
  if os.path.exists("training_images"):
    shutil.rmtree("training_images")

  os.mkdir("training_images")

  if condition==False:
    files=os.listdir("image_accurate")

  else:
    if os.path.exists("image_accurate_to_correct"):
      os.rename("image_accurate_to_correct", "image_accurate_corrected")
      files=os.listdir("image_accurate_corrected")

  cropped_aruco_list=os.listdir("cropped_aruco")
  for file in files:
      file=file[3:]
      for file_aruco in cropped_aruco_list:
        if file_aruco==file:
          shutil.copy2('cropped_aruco/'+file, 'training_images')  #copy with metadata
      #print('copying.. '+file+" to " + "training_images")


  if os.path.exists("deeplabcut_training.csv"):
    os.remove("deeplabcut_training.csv")



  new_data=v_data_n_toWrite.iloc[:3]

  for i in tqdm(files):
    i=i[3:]
    for k in range(0,v_data_n_toWrite.shape[0]):
      if  v_data_n_toWrite.iloc[k,2]==i:
        transposed_data=pd.DataFrame(v_data_n_toWrite.iloc[k,:]).transpose()
        new_data=pd.concat([new_data,transposed_data])

  new_data_to_write=new_data.copy()
  new_data_to_write.iloc[:,2] = new_data_to_write.iloc[:,2].str.replace(".jpg", '.png')
  new_data_to_write.to_csv(r'deeplabcut_training.csv', index=None)

  reduced_new_data=new_data.copy()
  reduced_size=200 # length of the reduced dataset that we want
  remove_n= len(reduced_new_data)- reduced_size-3
  drop_indices = np.random.choice(reduced_new_data.index.values[3:], remove_n, replace=False)
  v_data_reduced = reduced_new_data.drop(drop_indices)


  if os.path.exists("deeplabcut_training_reduced.csv"):
    os.remove("deeplabcut_training_reduced.csv")

  v_data_reduced_toWrite=v_data_reduced.copy()
  v_data_reduced_toWrite.iloc[:,2] = v_data_reduced_toWrite.iloc[:,2].str.replace(".jpg", '.png')
  v_data_reduced_toWrite.to_csv(r'deeplabcut_training_reduced.csv', index=None)


  if os.path.exists("reduced_training_images"):
    shutil.rmtree("reduced_training_images")

  os.mkdir("reduced_training_images")

      
  files_reduced=v_data_reduced.iloc[3:,2]


  for file in files_reduced:
      
      shutil.copy2('cropped_aruco/'+file, 'reduced_training_images')  #copy with metadata


if condition_cluster==True:

  files_clustered=os.listdir("clustered_training_images")

  if os.path.exists("clustered_training.csv"):
    os.remove("clustered_training.csv")

  new_data_clustered=v_data_n_toWrite.iloc[:3]

  for i in tqdm(files_clustered, desc="Writing clustered data: "):
    for k in range(0,v_data_n_toWrite.shape[0]):
      if  v_data_n_toWrite.iloc[k,2]==i:
        transposed_data=pd.DataFrame(v_data_n_toWrite.iloc[k,:]).transpose()
        new_data_clustered=pd.concat([new_data_clustered,transposed_data])

  new_data_clustered.iloc[:,2] = new_data_clustered.iloc[:,2].str.replace(".jpg", '.png')
  new_data_clustered.to_csv(r'deeplabcut_training_clustered.csv', index=None)

  


