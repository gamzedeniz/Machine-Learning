import csv
import cv2
import numpy as np
import os
import filter
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt

def save_gabor(img_dir,imglabel): #save the extracted 1d vectors from gabor to a csv file
    arrayfile=imglabel+".csv" #use label's name as the file name
    with open(arrayfile,mode ='a') as tmp:
        for img in os.listdir(img_dir):
            img2 = cv2.imread(os.path.join(img_dir, img)).astype(np.float32)
            size=(256,256)#before extracting the feature, resize the image to 256x256 to get equal shaped numpy arrays
            img2= cv2.resize(img2, size, interpolation = cv2.INTER_AREA)

            data=asarray([filter.Gabor_process(img2)])
            # save to csv file
            savetxt(tmp, data, delimiter=',')

    tmp.close()

def read_images(img_dir,imglabel):
    labels = ['COVID','NORMAL','Viral Pneumonia']
    global label
    global canny
    global tiny

    tmpcanny=[]
    tmptiny=[]
    tmplabel=[]
    for img in os.listdir(img_dir): #input_path
        img2 = cv2.imread(os.path.join(img_dir, img)).astype(np.float32)
        size=(256,256)#before extracting the feature, resize the image to 256x256 to get equal shaped numpy arrays
        img2= cv2.resize(img2, size, interpolation = cv2.INTER_AREA)
        #add each array to the temporary list
        tmpcanny.append(filter.Canny_edge(img2))
        tmptiny.append(filter.tinyimage(img2))
        tmplabel.append(labels.index(imglabel))
    #add list to global list for computations
    tiny+=tmptiny
    canny+=tmpcanny
    label+=tmplabel

def read_file(imglabel): #load the extracted 1d vectors from csv file
    labels = ['COVID','NORMAL','Viral Pneumonia']
    global gabor
    #global label
    #tmplabel=[]
    tmpgabor=[]
    arrayfile=imglabel+".csv"
    data = loadtxt(arrayfile, delimiter=',') #load from file, all rows into an array
    for i in range(data.shape[0]):
        tmpgabor.append(data[i]) #add each line into a temporary list
        #tmplabel.append(labels.index(imglabel))
        #if the labels are already stored with read images function then I won't use lines for storing labels in this function

    gabor+=tmpgabor #add list to global list for computations
    #label+=tmplabel

gabor=[]
canny=[]
tiny=[]
label=[]
#for saving each classes's gabor feature arrays
save_gabor("train\\NORMAL","NORMAL")
save_gabor("train\\COVID","COVID")
save_gabor("train\\Viral Pneumonia","Viral Pneumonia")

#for tiny and canny features extract arrays
read_images("train\\NORMAL","NORMAL")
read_images("train\\COVID","COVID")
read_images("train\\Viral Pneumonia","Viral Pneumonia")

#read from saved file to gabor list
read_file("NORMAL")
read_file("COVID")
read_file("Viral Pneumonia")


def crossvalidation (k,train,label,knn):#k fold cross validation, knn for k-NN
    block=round( len(train)/k ) #to divide train data into blocks
    Accy=0
    for i in range(k):
        if((len(train)%k > 0) and i==k-1): #if there is remainder, dont lose data
            x_testset=train[i*block:]
            y_testset=label[i*block:]
        else: #get the block as a test set
            x_testset=train[i*block:block*i+block]
            y_testset=label[i*block:block*i+block]
        starting_point=i*block #storing this to print out the falsely classified images
        x_trainset=train[block*i+i:] #take the rest of the blocks as train set
        y_trainset=label[block*i+i:]
        if(i!=0): #add into train set dont miss any data
            x_trainset.append(train[0:i*block])
            y_trainset.append(label[0:i*block])
        correct=0

        for i in range (len(x_testset)):#for sample in x_testset:
            #choosing only k-NN or weighted k-NN, it is up to which one I am looking for
            prediction=predict(x_trainset,y_trainset,x_testset[i],knn) #for knn
            #prediction=weightedprediction(x_trainset,y_trainset,x_testset[i],5) #for weighted knn
            if(prediction==y_testset[i]):
                correct+=1
            else:
                if(i%29==0): #just a random prime number
                    print("Falsely claffied:",starting_point+i,prediction,y_testset[i]) #see the wrong labeled ones
        Acc=100*correct/(len(x_testset))
        Accy+=Acc
        print("For k fold value",k,"For knn", knn,"Accuracy: ",Acc)
    print("Mean Acc",Accy/k)

def nearest_neighbors(train_set,trainlabel,test_sample):
    distances=[]#calculate distances from a test sample to every sample in a training set
    for i in range(len(train_set)):#for train_sample in train_set:
        dist=calculate_euclidean(train_set[i],test_sample)
        distances.append((trainlabel[i],dist))
    distances.sort(key=lambda x:x[1])#sort in ascending order, based on a distance value
    return distances


def calculate_euclidean(sample1,sample2): #calculate euclidean distance
    distance = np.linalg.norm(sample1 - sample2)
    return distance

def predict(xtrain_set,ytrain_set,xtest_sample,k):
    neighbors=[]
    distances=nearest_neighbors(xtrain_set,ytrain_set,xtest_sample)
    for i in range(k): #get first k samples's labels
        neighbors.append(distances[i][0])
    closestlabels=[sample for sample in neighbors] #get closest ones's labels
    prediction=max(closestlabels,key=closestlabels.count) #to predict
    return prediction

def weightedprediction(xtrainset,ytrainset,testsample,k): #weighted knn
    class0=0.0
    class1=0.0
    class2=0.0
    distances=nearest_neighbors(xtrainset,ytrainset,testsample)
    for i in range(k): #get first k samples's labels
        if(distances[i][0]==0): #sum for all the classes
            class0+=(1/distances[i][1]) #w=1/distance
        elif(distances[i][0]==1):
            class1+=(1/distances[i][1])
        else:
            class2+=(1/distances[i][1])
    if(class0>class1 and class0 > class2): #get sum of each class, find which biggest then predict the class
        predicted=0
    elif(class1>class0 and class1 > class2):
        predicted=1
    else:
        predicted=2
    return predicted


#Same tests that I run
print("tiny")

print(" 3 - 7")
crossvalidation(3,tiny,label,7) #crossfold and weighted or k-NN's k value
print(" 3 - 10")
crossvalidation(3,tiny,label,10) #crossfold
print(" 5 - 7")
crossvalidation(5,tiny,label,7) #crossfold
print(" 5 - 10")
crossvalidation(5,tiny,label,10) #crossfold

print("canny")

print(" 3 - 7")
crossvalidation(3,canny,label,7) #crossfold
print(" 3 - 10")
crossvalidation(3,canny,label,10) #crossfold


print("gabor")

print(" 3 - 7")
crossvalidation(3,gabor,label,7) #crossfold
print(" 3 - 10")
crossvalidation(3,gabor,label,10) #crossfold
print(" 5 - 7")
crossvalidation(5,gabor,label,7) #crossfold
print(" 5 - 10")
crossvalidation(5,gabor,label,10) #crossfold




