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
    
    #Crop all the images and store them in a sub-folder :
    #All our co-ordinate details for the images to be cropped around the human are in csv file.
    human_details = []
    file_det =open('final_data_csv.csv', 'rb')
    csv_f=csv.reader(file_det,delimiter=',')
    for row in csv_f :
        human_details.append(row)

    file_name_new=0
    human_ornot=[]
    count_total=0
    #New excel file to help with shuffling has image name and the label:
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet 1")
    
    #Now crop all the positive images and store them in a subfolder within the same path where the training images are there
    #for num_eg in xrange(0,len(human_details)-1) :
    for num_eg in xrange(0,2500) :
                file_name=human_details[num_eg][0]
                file_name=file_name.replace("'", "")
                write_split=os.path.splitext(file_name)[0]
                img_path=os.path.join(train_images_path,file_name)
                image_fromlist=cv2.imread(img_path)
                #cv2.imshow("Before_Cropping",image_fromlist)
                #cv2.waitKey(0)
                x_curr=float(human_details[num_eg][1])
                w_curr=float(human_details[num_eg][3])
                y_curr=abs(float(human_details[num_eg][2]))
                h_curr=float(human_details[num_eg][4])
                image_crop=image_fromlist[y_curr:(y_curr+h_curr),x_curr:(x_curr+w_curr)]
                #Resize the image
                #new_size = (400, int(image_crop.shape[0] * scale_factor))
                new_size =(500, 500)
                # perform the actual resizing of the image and show it
                resized_img = cv2.resize(image_crop, new_size, interpolation = cv2.INTER_AREA)
                #cv2.imshow("resized", resized_img)
                #cv2.waitKey(0)
                #file_name_new+=1
                image_crop_path=intermed_path + str(write_split) + '.jpg'
                #human_ornot.append(1)
                cv2.imwrite(image_crop_path,resized_img)
                sheet1.write(count_total,0,file_name)
                sheet1.write(count_total,1,1)
                count_total+=1
    print "Cropped and resized positive images"
    count_pos=count_total-1
    count_neg=0
    
    #We also need negative images, so we have to resize the image to the same size as positive images:        
    #for num_neg in glob.glob(os.path.join(neg_images_path)):
    for num_neg in os.listdir(neg_images_path):
                count_neg+=1
                if count_neg > 1500 :
                    break
                image_neg=cv2.imread(os.path.join(neg_images_path,num_neg))
                #Resize the image
                #scale_factor = 400.0 / image_neg.shape[1]
                #new_size = (400, int(image_neg.shape[0] * scale_factor))
                new_size = (500, 500)
                # perform the actual resizing of the image and show it
                resized_img = cv2.resize(image_neg, new_size, interpolation = cv2.INTER_AREA)
                #cv2.imshow("resized", resized)
                #cv2.waitKey(0)
                #file_name_new+=1
                #human_ornot.append(0)
                write_split=os.path.splitext(num_neg)[0]
                image_new_path=intermed_path + str(write_split) + '.jpg'
                cv2.imwrite(image_new_path,resized_img)
                write_name=os.path.splitext(num_neg)[0] + '.jpg'
                sheet1.write(count_total,0,write_name)
                sheet1.write(count_total,1,0)
                count_total+=1
    print "Resized negative images"
    count_neg=(count_total-1)-count_pos
    book.save("shuffling.xls")
