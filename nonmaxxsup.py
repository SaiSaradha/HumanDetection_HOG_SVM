#"Function to perform Non-maxima Suppression"
#import from params *
import numpy as np

def nonmaxsup(proposals):
    final_box=[]
    patch_size=([50,50])
    boundingbox_no=len(proposals)
    if boundingbox_no==0:
        return final_prop
    #Now we have n no of bounding boxes, out of which our goal is to suppress all of them except one"
    x2_val=[]
    y2_val=[]
    area_all=[]
    final_box=[]
    final_prop=[]
    conf_final=[]
    x1_final=[]
    y1_final=[]
    x2_final=[]
    y2_final=[]
    prop=zip(*proposals)
    x1_val=list(prop[0])
    y1_val=list(prop[1])
    conf_f=list(prop[2])
    print x1_val
    print y1_val
    print len(proposals)
    print conf_f
    max_ind=conf_f.index(max(conf_f))
    conf_max=conf_f[max_ind]
    thresh=0.2*conf_max
    
    for i in xrange(0,len(proposals)):
        #the x2 and y2 values are fixed (+ patch size) 
        x2_val.append(x1_val[i]+patch_size[0])
        y2_val.append(x2_val[i]+patch_size[1])
        #Now, lets eliminate the bounding boxes less than the threshold :
        if (conf_f[i] < thresh) :
            continue
        else :
            final_box.append([x1_val[i],y1_val[i],x2_val[i],y2_val[i]])
            conf_final.append([conf_f[i]])

    conf_final=list(conf_final)
    #Sort the bounding boxes based on the value of confidence scores:"
    #bbox stores the indices of the confidence scores in the ascending order
    bbox_sorted=list(np.argsort(conf_final))
    #index of the largest confidence score
    largest=len(bbox_sorted)-1
    #index of the box that has the maximum score
    max_box=bbox_sorted[largest]
    print "The bounding box with largest confidence value is {}".format(max_box)
    print "Index of max confidence score : " + str(largest)
    print max_box
    print final_box

    #Now pick the coordinates of the bounding box with max score :
    #find the index of maximum score in conf_f
    
    max_x1=x1_val[max_ind]
    max_y1=y1_val[max_ind]
    max_x2=x2_val[max_ind]
    max_y2=y2_val[max_ind]

    #Now let us try to find those bounding box co-ordinates that are within two boxes away from this coordinate :

    for i in xrange(0,len(final_box)):
        if x1_val[i] > ((1.5*patch_size[0])+max_x1) :
            continue
        else:
            x1_final.append([x1_val[i]])
        if y1_val[i] > ((1.5*patch_size[0])+max_y1):
            continue
        else:
            y1_final.append([y1_val[i]])
        if x2_val[i] > ((1.5*patch_size[0])+max_x2) :
            continue
        else:
            x2_final.append([x2_val[i]])
        if y2_val[i] > ((1.5*patch_size[0])+max_y2):
            continue
        else:
            y2_final.append([y2_val[i]])

    print x1_final
    print y1_final
    print x2_final
    print y2_final

    xf=min(x1_final)
    yf=min(y1_final)
    x2f=max(x2_final)
    y2f=max(y2_final)
    
    final_prop.append([xf[0],yf[0],x2f[0],y2f[0]])
    return final_prop
