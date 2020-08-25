import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import os
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import thin, skeletonize
import glob
import scipy.ndimage
import sys
np.set_printoptions(threshold=sys.maxsize)
from skimage import morphology
import time

def load_image(imagePaths ):
    with cbook.get_sample_data(imagePaths) as image_file:
        img = plt.imread(image_file)
    return img

def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3,3), 0)
    ret,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)  
    # kernel = np.ones((3,3), np.uint8) 
    # thresh = cv2.dilate(thresh, kernel, iterations=1)
    # canny = cv2.Canny(thresh,25,200)
    return thresh


def find_contour(thresh):
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def find_area(thresh,contours):
    if len(contours) != 0:
        cv2.drawContours(thresh, contours, -1, (255,255,255), -1)
        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        epsilon = 0.005*cv2.arcLength(c,True)
        #epsilon = 2000
        approx = cv2.approxPolyDP(c,epsilon,True)
        area_lis.append(round(cv2.contourArea(c)))
        hull = cv2.convexHull(c)
        
        # cv2.drawContours(thresh, c,0, 0, -1)
        cv2.drawContours(thresh, c, -1, (255, 255, 255), -1)
    return area_lis , thresh

def find_perimeter(contours):
    c = max(contours, key = cv2.contourArea)
    peri = round(cv2.arcLength(c,True))
    peri_lis.append(peri)
    return peri_lis

def valid_points(img):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    dilated = cv2.dilate(img, element)
    #cv2.imshow('Filled objects', dilated)
    img = dilated
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    return skel

def find_center(contour):
    c = max(contours, key = cv2.contourArea)
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    a = [cX, cY]
    center_lis.append(a)
    return center_lis
def find_length(thresh_image):
    thresh_image = thresh_image.astype(np.uint8)
    thined = thin(thresh_image)
    skel = skeletonize(thresh_image)
    #thined = thined.astype(np.uint8)
    skel = skel.astype(np.uint8)
    return skel,thined
def distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
def line(x1,y1,x2,y2):
    if x1 != x2:
        slope = (y1-y2)/(x1-x2)
        bias = y1 - slope*x1
    return slope,bias

def constrast_streching(img, r1, s1, r2, s2, r3 = 250.0, s3 = 250.0):
    img_gray = img.copy()
    img_gray2=np.where((img_gray<r1), np.floor(img_gray*(s1/r1)),
        np.where(((img_gray>r1)&(img_gray<r2)) , np.floor(s1+(img_gray-r1)*((s2-s1)/(r2-r1))),
            np.where(((img_gray>r2)&(img_gray<r3)) , np.floor(s2+(img_gray-r2)*((s3-s2)/(r3-r2))),img_gray))) 
    return img_gray2

def enhence(rgb_image):
    image_enhanced = rgb_image
    image_channel0 = image_enhanced[:,:,0]
    image_channel1 = image_enhanced[:,:,1]
    image_channel2 = image_enhanced[:,:,2]

    # img_gray=cv2.cvtColor(image_enhanced,cv2.COLOR_BGR2GRAY)

    image_channel0_en = constrast_streching(image_channel0, 100, 60, 160, 200, r3 = 255.0, s3 = 255.0)
    image_channel1_en = constrast_streching(image_channel1, 100, 60, 160, 200, r3 = 255.0, s3 = 255.0)
    image_channel2_en = constrast_streching(image_channel2, 100, 60, 160, 200, r3 = 255.0, s3 = 255.0)

    image_enhanced[:,:,0] = image_channel0_en
    image_enhanced[:,:,1] = image_channel1_en
    image_enhanced[:,:,2] = image_channel2_en
    return image_enhanced

def save_to_csv(number_lis,area_lis,peri_lis,length_lis,x_center_lis,y_center_lis,path_save):
    data = pd.DataFrame({'ID': number_lis,
                         'area': area_lis,
                         'peri' : peri_lis,
                         'length':length_lis,
                         'x_center':x_center_lis,
                         'y_center':y_center_lis})
    data.to_csv(path_save, index = False)

data_path ="your path\\Data\\image_tail_off"
data_path =data_path.replace('\\','\\\\')
img_list = glob.glob(os.path.join(data_path,"*.jpg"))

number = 0
area_lis = []
number_lis = []
peri_lis = []
length_lis = []
x_center_lis = []
y_center_lis = []

for image_path in  img_list :
    st = time.time()
    img = cv2.imread(image_path)
    image_enhance = enhence(img)
    # binary image
    thresh = image_processing(image_enhance)
    thresh1 = thresh.astype(bool) 
    # cleaned : only shrimp image 
    cleaned = morphology.remove_small_objects(thresh1, min_size=200, connectivity=1,in_place =True)
    cleaned = cleaned.astype(float)
    cleaned = cleaned*255
    cleaned = np.uint8(cleaned)
    
    cleaned = cleaned/255
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (25,25))
    eroded = cv2.erode(cleaned,element)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, element)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, element)
    cleaned = np.uint8(cleaned)

    # find contours
    contours = find_contour(cleaned)
    # area
    area_lis , image = find_area(cleaned,contours)
    en = time.time()
    # max contours to approximate contour
    c = max(contours, key = cv2.contourArea)
    epsilon = 0.01*cv2.arcLength(c,True)
    # new_image: image with only approximate contour
    new_image = np.zeros_like(cleaned)
    approx = cv2.approxPolyDP(c,epsilon,True)
    cv2.drawContours(new_image, [approx], -1, (255, 255, 255), -1)
    new_image = new_image/255
    # skel1 : image after skeleton
    skel1 = skeletonize(new_image)
    skel1 = np.float32(skel1)
    skel2 = np.zeros_like(cleaned)
    skel1 =skel1 *255

    height_y,width_x = skel1.shape
    x_skel =[]
    y_skel =[]

    for i in range(height_y):
        for j in range(width_x):
            if(skel1[i][j] == 0 ):
                
                skel2[i][j] = 0 
            else:
                x_skel.append(j)
                y_skel.append(i)
                skel2[i][j] = 255

    skel2 = np.uint8(skel2)
    x,y,w,h = cv2.boundingRect(c) 

    peri_lis = find_perimeter(contours)

    central_box_x = round(x + w/2)
    central_box_y = round(y + h/2)

    x_skel_center = round(x+w/2)
    x_center_lis.append(x_skel_center)
    x_skel = np.array(x_skel)
    y_skel = np.array(y_skel)
    x_sub_skel = []
    y_sub_skel = []

    index_skel = np.where((round(x_skel_center - w/3.7) <= x_skel )& (round(x_skel_center + w/2.8) >= x_skel))

    index_skel = index_skel[0].tolist()
    y_skel_center = y_skel[np.where(x_skel == x_skel_center)[0][0]]
    y_center_lis.append(y_skel_center)
    for k in range (len(index_skel)):
        x_sub_skel.append(x_skel[index_skel[k]])
        y_sub_skel.append(y_skel[index_skel[k]]) 
    # print(x_sub_skel)
    # print(y_sub_skel)
    
    sub_skel_image = np.zeros_like(cleaned)
    for t in range (len(x_sub_skel)):
        sub_skel_image[y_sub_skel[t]][x_sub_skel[t]] = 255
    
    x_sub_skel = np.array(x_sub_skel)
    x_end_sub_skel_max = np.amax(x_sub_skel)
    x_end_sub_skel_min = np.amin(x_sub_skel)
    
    x_end_sub_skel_max_index =  np.where(x_sub_skel == np.amax(x_sub_skel))[0][0]
    x_end_sub_skel_min_index =  np.where(x_sub_skel == np.amin(x_sub_skel))[0][0]
    y_end_sub_skel_max =  y_sub_skel[x_end_sub_skel_max_index]
    y_end_sub_skel_min =  y_sub_skel[x_end_sub_skel_min_index]
   
    x_another_min = x_end_sub_skel_min + 10
    y_another_min = y_sub_skel[np.where(x_sub_skel == x_another_min)[0][0]]
    x_another_max = x_end_sub_skel_max - 15
    y_another_max = y_sub_skel[np.where(x_sub_skel == x_another_max)[0][0]]

    slope_min,bias_min = line(x_end_sub_skel_min,y_end_sub_skel_min,x_another_min,y_another_min)
    slope_max,bias_max = line(x_end_sub_skel_max,y_end_sub_skel_max,x_another_max,y_another_max)
    
 
    for p in range(len(c)):
        if(c[p][0][0] < x_end_sub_skel_min):
            if (abs(c[p][0][0]*slope_min+bias_min - c[p][0][1]) < 3):
                x_full_skel_min = c[p][0][0]
                y_full_skel_min = c[p][0][1]
                break
    for p in range(len(c)):
        if(c[p][0][0] > x_end_sub_skel_max):
            if (abs(c[p][0][0]*slope_max + bias_max - c[p][0][1]) < 3):
                x_full_skel_max = c[p][0][0]
                y_full_skel_max = c[p][0][1]
                break
    # sub_skel_image = cv2.putText(sub_skel_image, '^', (x_another_min,y_another_min),cv2.FONT_HERSHEY_SIMPLEX ,1, (255,255,255), 2, cv2.LINE_AA) 
    # sub_skel_image = cv2.putText(sub_skel_image, '^', (x_another_max ,y_another_max ),cv2.FONT_HERSHEY_SIMPLEX ,1, (255,255,255), 2, cv2.LINE_AA) 
    # sub_skel_image = cv2.putText(sub_skel_image, '*', (x_end_sub_skel_min,y_end_sub_skel_min),cv2.FONT_HERSHEY_SIMPLEX ,1, (255,255,255), 2, cv2.LINE_AA) 
    # sub_skel_image = cv2.putText(sub_skel_image, '*', (x_end_sub_skel_max,y_end_sub_skel_max),cv2.FONT_HERSHEY_SIMPLEX ,1, (255,255,255), 2, cv2.LINE_AA) 
    # sub_skel_image = cv2.putText(sub_skel_image, '-', (x_full_skel_min,y_full_skel_min ),cv2.FONT_HERSHEY_SIMPLEX ,1, (255,255,255), 4, cv2.LINE_AA) 
    # sub_skel_image = cv2.putText(sub_skel_image, '-', (x_full_skel_max,y_full_skel_max),cv2.FONT_HERSHEY_SIMPLEX ,1, (255,255,255), 4, cv2.LINE_AA) 
    
    sum_length = 0
    
    index_temp_sort = sorted(range(len(x_sub_skel)), key=lambda ko: x_sub_skel[ko])
    x_sub_skel = np.sort(x_sub_skel)
    # print(index_temp_sort)
    # print(x_sub_skel)
    y_sub_skel_new =[]
    for z in range (len(y_sub_skel)):
        y_sub_skel_new.append(y_sub_skel[index_temp_sort[z]])
    # print(y_sub_skel_new)
    for q in range (len(y_sub_skel) - 1 ):
        sum_length = sum_length + distance(x_sub_skel[q],y_sub_skel_new[q],x_sub_skel[q+1],y_sub_skel_new[q+1])
    total_length = sum_length + distance(x_end_sub_skel_min,y_end_sub_skel_min,x_full_skel_min,y_full_skel_min) + distance(x_end_sub_skel_max,y_end_sub_skel_max,x_full_skel_max,y_full_skel_max)
    # print(total_length)
    total_length = round(total_length,4)
    length_lis.append(total_length)
    number_lis.append(number)
    en =time.time()
    print('total time',en-st)
    save_to_csv(number_lis,area_lis,peri_lis,length_lis,x_center_lis,y_center_lis,"your path\\Get_feature.csv")   
    number +=1
    cv2.waitKey(0)
