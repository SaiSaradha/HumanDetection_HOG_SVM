#Main file to test the human detection

import os
import cv2
import cv2.cv as cv_hog
import scipy.io
import csv
import glob
import hog_extraction
import timeit
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from skimage.feature import hog
from skimage.io import imread
import xlwt
import random
import openpyxl

if __name__ =="__main__":
    #The procedure to go about this human detection is as follows :
    #   1. Get the images
    #   2. Extract the hog features
    #   3. Train the Linear SVM classifier
    #   4. Test the classifier on test images

    #1. Get the training and test images :
    train_images_path = "../images/training_images"
    neg_images_path = "../images/neg_images"
    crop_images_path="../images/crop/"
    test_images_path = "../images/test_images"
    svm_model_path= "../SVM_Model/svm_class.model"
    intermed_path="../images/intermed/"

    #Import the annotations containing the bounding box and the label for the training images:
    #annot_list=scipy.io.loadmat('./image_data_final.mat')
    #human_ornot=scipy.io.loadmat('./actual_humanpresent.mat')

    #Extract the hog features for training the classfier:
    #Here, we have two sets of images, positive images with label 1 and negative images with label 0
    #Positive images contain the human while negative does not. For positive images, we send the image
    #within the bounding box (obtained from annotattions) only.
    hog_feat_total=[]
    file_name_new=0
    human_ornot=[]
    count_total=0

    human_det_pos_neg = []
    file_det1 =open('shuffling_csv_f.csv', 'rU')
    csv_f1=csv.reader(file_det1,delimiter=',')
    for val in csv_f1:
        human_det_pos_neg.append(val)

    count_total=len(human_det_pos_neg)  
    #read row numbers
    row_num=[]
    random.seed(12345)
    row_num.append(random.sample(range(0,count_total-1),(count_total-1)))
    
    for shuff in row_num[0]:
                file_name=human_det_pos_neg[shuff][0]
                file_name=file_name.replace("'", "")
                img_path=os.path.join(intermed_path,file_name)
                image_fromlist=cv2.imread(img_path)
                #cv2.imshow('image',image_fromlist)
                #cv2.waitKey(0)
                file_name_new+=1
                image_crop_path=crop_images_path + str(file_name_new) + '.jpg'
                human_ornot.append(human_det_pos_neg[shuff][1])
                cv2.imwrite(image_crop_path,image_fromlist)
    
    hog_feat_total=[]
    hog_feat_curr=[]
    count_feat=0
    start=timeit.default_timer()
    #for images_seq in sorted (glob.glob(os.path.join(crop_images_path))):
    for images_seq in sorted (os.listdir(crop_images_path)):
        if not images_seq.startswith('.') :
            img_curr=imread(os.path.join(crop_images_path,images_seq),as_grey=True)
            hog_feat_curr = hog(img_curr, 9, [8,8], [2,2], False, True)
            hog_feat_total.append(hog_feat_curr)
    print "Extracted hog features for all images "
    stop=timeit.default_timer()
    print "Run time for hog extraction  " + str(stop-start)
    
    #Display the size of hog_feat (it should be examples_no x no_features) and labels size
    feat_noexmp=len(hog_feat_total)
    feat_nofeat=len(hog_feat_total[0])
    labels_size=len(human_ornot)
    print " Number of images :  " + str(feat_noexmp)
    print " Number of features per image :  " + str(feat_nofeat)
    print "Label size : " +str(labels_size)

    #Training the SVM Classifier :
    start=timeit.default_timer()
    clf=LinearSVC()
    print "Training the SVM Classifier"
    clf.fit(hog_feat_total,human_ornot)
    #Path to store the SVM model that we have built now :
    #if not os.path.isdir(svm_model_path) :
     #   os.makedirs(svm_model_path)    
    joblib.dump(clf, svm_model_path)
    print "SVM successfully trained"
    stop=timeit.default_timer()
    print "Time for SVM Training :  " + str(stop-start)
