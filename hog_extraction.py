#"Code to extract HOG features and visualize them"
import cv2.cv as cv
import numpy as np

#"Function to compute integral image from bin image:"
def integral_img (image,bins):
    #numorient=bins;
    size = cv.GetSize(image)
    gray_img = cv.CreateImage(size, 8, 1)
    xsobel = cv.CreateImage(size, 32, 1)
    ysobel = cv.CreateImage(size, 32, 1)
    
 #    "RGB to grayscale conversion"
    cv.CvtColor(image, gray_img, cv.CV_BGR2GRAY)	
    
  #  "Gradient computation using sobel operators"
    cv.Sobel(gray_img, xsobel, 1, 0, 3)
    cv.Sobel(gray_img, ysobel, 0, 1, 3)

    integral_hog= np.zeros((size[1], size[0], bins))
    for y in xrange(0, size[1]-1):
        for x in xrange(0, size[0]-1):
            angle1= (int(round(180*np.arctan2(ysobel[y,x], xsobel[y,x])/np.pi))+ 180)
            if angle1>=0 and angle1<40:
                angle=0
            elif angle1>=40 and angle1<80:
                angle=1
            elif angle1>=80 and angle1<120:
                angle=2
            elif angle1>=120 and angle1<160:
                angle=3
            elif angle1>=160 and angle1<200:
                angle=4
            elif angle1>=200 and angle1<240:
                angle=5
            elif angle1>=240 and angle1<280:
                angle=6
            elif angle1>=280 and angle1<320:
                angle=7
            elif angle1>=320 and angle1<=360:
                angle=8
            grad_m = np.sqrt(xsobel[y,x]*xsobel[y,x]+ysobel[y,x]*ysobel[y,x])
            integral_hog[y,x,angle] += grad_m
            
 #       "Now we build the integral image"
    for x in xrange(1, size[0]-1):
        for orient in xrange(bins):
            integral_hog[y,x,orient] += integral_hog[y,x-1,orient]
    for y in xrange(1, size[1]-1):
        for orient in xrange(bins):
            integral_hog[y,x,orient] += integral_hog[y-1,x,orient]
    for y in xrange(1, size[1]-1):
        for x in xrange(1, size[0]-1):
            for orient in xrange(bins):
                integral_hog[y,x,orient] += integral_hog[y-1,x,orient] + integral_hog[y,x-1,orient] - integral_hog[y-1,x-1,orient]
    return integral_hog


def integral_hog_window(image_h, rect_val):
    bins = image_h.shape[2]
    hog_features = np.zeros(bins)
    for orient in xrange(bins):
        hog_features[orient] = image_h[rect_val[1],rect_val[0],orient] + image_h[rect_val[3],rect_val[2],orient] - image_h[rect_val[1],rect_val[2],orient] - image_h[rect_val[3],rect_val[0],orient]
    return hog_features

def hog_visualize(hog_image_v, integral_img, cell):
    hog_feat= [0] * 9
    width_img,height_img = cv.GetSize(hog_image_v)
    halfcell = cell/2
    num_cells_w,num_cells_h = width_img/cell,height_img/cell
    norient = integral_img.shape[2]
    mid = norient/2
    for y in xrange(num_cells_h-1):
        for x in xrange(num_cells_w-1):
            px,py=x*cell,y*cell
            #features = integral_hog_window(integral_img, (px,py,max(px+8, width_img-1),max(py+8, height_img-1)))
            features = integral_hog_window(integral_img, (px, py, px+cell, py+cell))
            hog_feat = hog_feat + list(features)
            px += halfcell
            py += halfcell

            
            #L1-norm, nice for visualization
            total = np.sum(features)
            maximum_value_feature = np.max(features)
            if total > 1e-3:
                normalized = features/maximum_value_feature
                N = norient
                final = []
                for i in xrange(N):
                    maximum_orient = normalized.argmax()
                    valmax = normalized[maximum_orient]
                    x1 = int(round(valmax*halfcell*np.sin(np.deg2rad(45*(maximum_orient-4)))))
                    y1 = int(round(valmax*halfcell*np.cos(np.deg2rad(45*(maximum_orient-4)))))
                    gradient_val = int(round(255*features[maximum_orient]/total))
                    #print "values of x1 =",x1,"and y1=",y1, "and gv=",gradient_val

                    #don't draw if less than a threshold
                    if gradient_val < 30:
                        break
                    final.insert(0, (x1,y1,gradient_val))
                    normalized[maximum_orient] = 0.
                    
                #draw from smallest to highest gradient magnitude
                for i in xrange(len(final)):
                    x1,y1,gradient_val = final[i]
                    cv.Line(hog_image_v, (px-x1,py+y1), (px+x1,py-y1), cv.CV_RGB(gradient_val, gradient_val, gradient_val), 1, 8)
            else:
                #don't draw if there's no reponse
                pass
    return hog_feat
