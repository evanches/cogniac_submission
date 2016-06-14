import cv2
import numpy as np
import os

orig_dir = 'original_images'
mod_dir = 'modified_images'


result = open('results.csv', 'w')

orig_sift_points = []
test = True

sift = cv2.xfeatures2d.SIFT_create()


bf = cv2.BFMatcher(cv2.NORM_L2)

# Get SIFT points for original image
for file_name in os.listdir(orig_dir):
    print file_name
    img = cv2.imread(orig_dir + '/' + file_name)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8, copy=False)
   
    kp, des = sift.detectAndCompute(gray,None)
    tup = (file_name, kp, des)
    orig_sift_points.append(tup)

    print len(des)
    if test:
        test = False


# Get SIFT points for all modified images
for file_name in os.listdir(mod_dir):
    print "Modified"
    print file_name
    img = cv2.imread(mod_dir + '/' + file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8, copy=False)

    kp, des = sift.detectAndCompute(gray,None)
    tup = (file_name, kp, des)

      
    #Go through all original SIFT points
    for orig_file in orig_sift_points:
        distances = []
        dist_test = 0
        matches = bf.match(des, orig_file[2])
        matches = sorted(matches, key = lambda x:x.distance)
        for i in range(25):
            dist_test = dist_test + matches[i].distance
        distances.append(dist_test)

    #Find the minimum of the distances
    min_index = distances.index(min(distances))

    print "Matched file is " +  str(orig_sift_points[min_index][0])
    result.write(orig_sift_points[min_index][0] + ',' +  file_name + '\n')    

