#Main file to test the human detection

import os
import cv2
import cv2.cv as cv_hog
import scipy.io
import csv
import glob
import hog_extraction
import timeit
import nonmaxxsup
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from skimage.io import imread
from skimage.feature import hog

#Now let us test the classifier on the unseen images :"
    #Using sliding window approach :
def image_patch_fun(image_f, patch_size, slide_size):
    for yval in xrange(0,image_f.shape[0], slide_size[1]):
           for xval in xrange(0,image_f.shape[1],slide_size[0]):
                yield(xval, yval, image_f[yval:yval+patch_size[1], xval:xval+patch_size[0]])

patch_size=([200,200])
step_size=([150,150])
proposals=[]
test_images_path = "../images/test_images"
svm_model_path= "../SVM_Model/svm_class.model"
clf=joblib.load(svm_model_path)
     
if __name__=="__main__" :               
    for images_testseq in os.listdir(test_images_path):
        if not images_testseq.startswith('.') :
           image_testcurr=imread(os.path.join(test_images_path,images_testseq),as_grey=True)
           h=image_testcurr.shape[1]
           w=image_testcurr.shape[0]
           new_size =(500, 500)
           image_testcurrnew = cv2.resize(image_testcurr, new_size, interpolation = cv2.INTER_AREA)
           count=0
           for (xval, yval, image_patch_n) in image_patch_fun(image_testcurrnew, patch_size, step_size):
                 #if xval > w or yval > h:
                  #      break
                 print xval
                 print yval
                 #Extract the hog feature of this image:
                 # perform the actual resizing of the image and show it
                 resized_img = cv2.resize(image_patch_n, new_size, interpolation = cv2.INTER_AREA)
                 hog_feat_test = hog(resized_img, 9, [8,8], [2,2], False, True)
                 svm_class=0
                 svm_class=clf.predict(hog_feat_test)
                 print svm_class
                 count+=1
                 if svm_class=='1':
                     proposals.append([xval, yval, float(clf.decision_function(hog_feat_test))])
                     #proposals.append([xval, yval])
                     print "Confidence Function" + str(clf.decision_function(hog_feat_test))
           #Image with all proposals as bounding boxes :
           img_prop=image_testcurrnew.copy()
           for (w_i, h_i, decf) in proposals:
               cv2.rectangle(img_prop,(w_i, h_i), (w_i+patch_size[1], h_i+patch_size[0]),(255,0,0))
           cv2.imshow("Potential_humans_in_image", img_prop)
           cv2.waitKey()

           #Using Non-Maxima Suppression to find only the human:
           final_human=nonmaxxsup.nonmaxsup(proposals)
           print "Final Human coordinates"
           print list(final_human)

           #Final human detected in the image:
           for (w_i, h_i, w2, h2) in final_human:
               cv2.rectangle(image_testcurrnew,(w_i, h_i), (w2, h2),1,8,0)
           cv2.imshow("Final - humans in the image", image_testcurrnew)
           cv2.waitKey()                  
