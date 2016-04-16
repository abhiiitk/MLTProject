################################################################################
##    Written By:   Abhishek Srivastava (12033)   and  Aditya Raj(12049)      ##
##                                 Group 24                                   ##
################################################################################

import cv2
import pandas as pd
from skimage.feature import hog
import pickle
import os,sys
import numpy as np
import vision
## Declaring SIFT thresholds (after TUNING)
sift = cv2.SIFT(contrastThreshold=0.08, edgeThreshold = 4)
##Declaring Objects 
trainkmeans = np.array([])
resizedimagesize=100
AspectR = np.array([])
features=[]
Label = np.array([], dtype = int)
imgcount = np.array([], dtype = int)
no_keypoints=[]
## Changing Directory to directory containg Video and Text Files (Text files
##contain Bounding Boxes of Vatic)
os.chdir('/media/abhishek/TOSHIBA EXT/videos')


## Function to Process the text files using Pandas library and numpy and get
## Labels, Box Size, Aspect Ratio etc

def maketable(filename, takeimages = 10):
    texdic1 = pd.read_csv(filename, sep = " ", header = None)

    texdic1.columns = ["Track_ID", "xmin", "ymin", "xmax", "ymax", "frame", \
                       "lost","occluded", "generated", "label"]
    texdic2 = texdic1.copy()
    texdic1['height'] = (texdic1.ymax - texdic1.ymin)
    texdic1['width'] = (texdic1.xmax - texdic1.xmin)
    # texdic1['Area'] = (texdic1.height*texdic1.width)
    texdic1['AspectR'] = texdic1.width/(texdic1.height + 0.0)
    temp = ((texdic1.occluded == 0) & (texdic1.lost == 0) & (texdic1.label != \
                                                             "Number-plate"))
    texdic1 = texdic1[temp]
    helprand  = np.bincount(texdic1.Track_ID)
    helprand2 = takeimages* (helprand >takeimages) + \
                (helprand <takeimages)*1*helprand
    cumsum = np.arange(1)
    cumsum = np.append(cumsum, helprand.cumsum())
    rownumber = np.array([], dtype = int)
    for i in np.arange(helprand.shape[0]):
        if helprand[i] > 0:
            rownumber = np.append(rownumber,np.random.choice(helprand[i], size =\
                            helprand2[i], replace = False, p= None) + cumsum[i])
    rownumber = np.sort(rownumber)
    texdic1.index = np.arange(texdic1.shape[0])
    readfin1 = texdic1.loc[rownumber]
    readfin1 = readfin1.sort(columns = ['frame'])
    readfin1.index = np.arange(readfin1.shape[0])
    return readfin1
##Function to Get Numerical Categorical Labels from Labels in text

def givelabel(string):
	if string=="Person":
		return 0
	elif string=="Motorcycle":
		return 1
	elif string=="Car":
		return 2
	elif string=="Bicycle":
		return 3
	elif string=="Rickshaw":
		return 4
	elif string=="Autorickshaw":
		return 5
	else: 
		return 6
##Fucntion to call the above function for text vector
def veclabel(strvec):
    a = 0*np.arange(len(strvec))
    for i in range(len(strvec)):
        a[i] = givelabel(strvec[i])
    return a

##Declaring Video Names and Names of textfiles containg the bounding box info
VideoNames = [ "datasample1.mov", "dec21h1330.dav",  "input_video_sample1.mov", \
               "input_video_sample2.mov", "input_video_sample3.mov", \
               "nov92015-1.dav", "nov92015-2.dav", "videosample5.mov" ]

TextNames = [ "datasample1.txt", "dec21h1330.txt",  "input_video_sample1.txt",\
              "input_video_sample2.txt", "input_video_sample3.txt",\
              "nov92015-1.txt", "nov92015-2.txt", "videosample5.txt" ]


## Declaring which will be output of maketable function on all text files
t1 = []

tabtab = [t1, t1, t1, t1, t1, t1, t1, t1]
## Reading Videos Frame by Frame, taking the boxes from relevant frames and then
## Extracting HOG features and SIFT keypoints from grayscale resized version of
## the image box

for k in range(8):
    table1 = maketable(TextNames[k])
    tabtab[k] = table1.copy()
    Label = np.append(Label, veclabel(tabtab[k].label))
    AspectR = np.append(AspectR,tabtab[k].AspectR)
    imgcount = np.append(imgcount, len(tabtab[k]))
    cap = cv2.VideoCapture(VideoNames[k])
    count = 0
    fcount = 0
    lentable = len(table1)
    while (count < tabtab[k].frame.max()+1):
        ret, frame = cap.read()
        while(table1.frame[fcount] == count):
            
            #crop_img = img[ytl:ybr,xtl:xbr]
            crop_img = frame[table1.ymin[fcount]:table1.ymax[fcount],\
                             table1.xmin[fcount]:table1.xmax[fcount]]
            gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray_img, None)
            no_keypoints.append(len(kp))
            if(len(kp)!=0):
                trainkmeans=np.append(trainkmeans,np.ravel(des))
            resized_img=cv2.resize(gray_img,(resizedimagesize,resizedimagesize))
            
            fd = hog(resized_img, orientations=9, pixels_per_cell=(10, 10), \
                     cells_per_block=(2, 2), visualise=False)
            features.append(fd)

            print fcount
            
            fcount = fcount + 1
            if(fcount == lentable):
                break

        count = count+1
        if k == 27:
            break

## 
imgind2 = imgcount.cumsum()
imgind1 = np.append(0,imgind2)[:k+1]

AspectR = AspectR.reshape(len(AspectR),1)

## Saving the variables for further use and deleting bigger ones to free RAM
with open('features.pickle', 'w') as f:
    pickle.dump([features], f)
with open('tables.pickle', 'w') as f:
    pickle.dump([tabtab], f)
with open ('kmeansdata.pickle','w') as f:
    pickle.dump([trainkmeans, no_keypoints],f)
with open ('data.pickle', 'w') as f:
    pickle.dump([imgind1, imgind2, Label,AspectR],f)

del features                         ### to free memory

## Commands to reload these variable in future if needed 


##with open('tables.pickle') as f:
##    [tabtab] = pickle.load(f)
##    
##with open ('kmeansdata.pickle') as f:
##    [trainkmeans,no_keypoints] = pickle.load(f)
##trainkmeans=trainkmeans.reshape(trainkmeans.shape[0]/128,128)
##
##with open ('data.pickle') as f:
##    [imgind1, imgind2, Label,AspectR] = pickle.load(f)

## Converting to numpy array
no_keypoints = np.array(no_keypoints)

## Taking a smaller subset (ranomized) from the entire SIFT keypoints coming
## from train data (fist seven videos out of which five are small and tow ar big


trainkmeans1 = trainkmeans[np.random.choice(sum(no_keypoints[:imgind1[7]]), 150000, \
                                            replace = False, p = None)]

del trainkmeans     ### to free memory, Deleting the bigger objects

## Traing Kmeans Cluster of 50 and 100 clusters (Shold be trained with higher
## Number of clusters but we used 100 due to limitations on Processor, RAM and
##Time capacity

kmeans_cluster50 = KMeans(n_clusters=50, random_state=1).fit(trainkmeans1)
kmeans_cluster100 = KMeans(n_clusters=100, random_state=1).fit(trainkmeans1)

del trainkmeans1                              ### to free memory

## Saving K-means Models

with open ('kmeansITR50.pickle','w') as f:
    pickle.dump([kmeans_cluster50],f)
with open ('kmeansITR100.pickle','w') as f:
    pickle.dump([kmeans_cluster100],f)


## Loading Earlier Deleted SIFT Keypoints

with open ('kmeansdata.pickle') as f:
    [trainkmeans,no_keypoints] = pickle.load(f)
trainkmeans=trainkmeans.reshape(trainkmeans.shape[0]/128,128)

## Commands to reload kmeans fitted models in future if needed 
##with open ('kmeansITR50.pickle') as f:
##    [kmeans_cluster50] = pickle.load(f)
##with open ('kmeansITR100.pickle') as f:
##    [kmeans_cluster100] = pickle.load(f)

## Predicting Cluster using Fitted Kmeans models 
kmeans_model50=kmeans_cluster50.predict(trainkmeans)
kmeans_model100 = kmeans_cluster100.predict(trainkmeans)

del trainkmeans                             ### to free memory

## Generating Bag of Word Features from SIFT keypoints (cluster labels)
count=0
sift_features50=[]
    
for j in range(0,len(Label)):
	sift_feature=[0]*50
	for t in range(0,no_keypoints[j]):
		sift_feature[kmeans_model50[count]]=sift_feature[kmeans_model50[count]]+1
		count=count+1
	sift_features50.append(sift_feature)

count=0
sift_features100=[]
    
for j in range(0,len(Label)):
	sift_feature=[0]*100
	for t in range(0,no_keypoints[j]):
		sift_feature[kmeans_model100[count]]=sift_feature[kmeans_model100[count]]+1
		count=count+1
	sift_features100.append(sift_feature)

## Connvering to numpy array
sift_features100  = np.array(sift_features100)
sift_features50 = np.array(sift_features50)

## Loading HOG Features
with open('features.pickle' ) as f:
    [features] = pickle.load(f)
features = np.array(features)

## Saving the necessary variables for use training testing purpose
with open ('KAMKADATA.pickle','w') as f:
        pickle.dump([imgind1, imgind2, features, sift_features100, \
                     sift_features50, AspectR, Label], f)

