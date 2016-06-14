import cv2
import numpy as np

file_name = 'validation_set.csv'
orig_dir = 'original_images/'
mod_dir = '~/modified_images'

sift = cv2.xfeatures2d.SIFT_create()
f = open(file_name, 'r')
bf = cv2.BFMatcher(cv2.NORM_L2)

for line in f:
    spl = line.split(',')
    print spl[0]
    print spl[1]
    print orig_dir + spl[1]
    img = cv2.imread(orig_dir + spl[1], 0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = img.asType(np.uint8, copy=False)

    img2 = cv2.imread(mod_dir + '/' + spl[0])
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    gray2 = gray2.astype(np.uint8, copy=False)


    kp1, des1 = sift.detectAndCompute(gray,None)

    kp2, des2 = sift.detectAndCompute(gray2,None)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    dist_test = 0

    for i in range(25):
        dist_test = dist_test + matches[i].distance


    print "Distance between match is: " + str(dist_test)
