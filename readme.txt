readme.txt for execution of EECS 440 Project - Human Detection in images using HoG and LinearSVM.

1. Test.py :  To run the test on the SVM model,  give the path of folder where test images are present in test_images_path variable. Execute this python file. It will predict the human in the image by drawing bounding box around it.
2. Shuffling_final.py: It takes the shuffled training set. Then it calls OpenCV built in hog for calculation of hog features. This hog feature is given into the Linear SVM for training. Though the hog_extraction.py written by us also returns the same hog features but it is quite slow as compared to built in hog of opencv. To train SVM with 4000 images, hog_extraction.py would have taken much longer time so we used the available function.
3. Final_detection.py: It is same as shuffling_final.py except it trains the SVM without shuffled examples.
4. Hog_extractiion.py : It takes the input image from final_detection.py and returns the calculated hog features to final_detection.
5. Nonmaxxsup.py:  It is called from final_detection.py. It takes the coordinates of proposed boxes by final_detection.py and returns single box which contains all the region of proposed boxes with confidence score greater than threshold.
6. Shuffling_csv_f.csv: It contains the annotations of positive and negative samples shuffled randomly.
7. Svm_class_old.model:  The generated svm model trained with 4000 sample images.
