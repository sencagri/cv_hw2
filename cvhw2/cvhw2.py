import cv2
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk
import numpy as np

def onCannyParamChange(x):
    pass

def main():
    img = cv2.imread("istaka4.jpg")
    kernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
    #img = cv2.filter2D(img, -1, kernel)
    img_o = img.copy()
    img = cv2.resize(img, None, None, 0.8, 0.8, cv2.INTER_CUBIC)
    hsv = img.copy()
    hsv2 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thres = cv2.threshold(gray, 127,255, cv2.THRESH_OTSU)
    img = gray.copy()
    img = cv2.equalizeHist(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]],np.uint8)

    #img = cv2.medianBlur(img, 3)
    r = 5
    selem = disk(r)
    
    cv2.namedWindow("Canny Edges")
    cv2.createTrackbar("cannyMin\n1", "Canny Edges", 100, 1000, onCannyParamChange)
    cv2.createTrackbar("cannyMax\n1", "Canny Edges", 200, 1000, onCannyParamChange)
    cv2.createTrackbar("sizefactor\n1", "Canny Edges", 100, 100, onCannyParamChange)

    adaptive_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    adaptive_img = cv2.bilateralFilter(adaptive_img, 9, 75,75)

    local_otsu_threshold = rank.otsu(img,selem)
    local_bin_img = (img >= local_otsu_threshold) * 255

    global_otsu_threshold = threshold_otsu(img)
    global_bin_img = (img >= global_otsu_threshold) * 255

    global_bin_img = global_bin_img.astype(np.uint8)
    #global_bin_img = cv2.erode(global_bin_img, kernel, iterations=8)
    #global_bin_img = cv2.dilate(global_bin_img, kernel, iterations=4)
    local_bin_img = local_bin_img.astype(np.uint8)
    #local_bin_img = cv2.erode(local_bin_img, kernel, iterations=8)
    #local_bin_img = cv2.morphologyEx(local_bin_img, cv2.MORPH_OPEN, kernel)
    #local_bin_img = cv2.medianBlur(local_bin_img, 11)
    #local_bin_img = cv2.medianBlur(local_bin_img, 3)

    #cv2.imshow("Original Image", img_o)
    #cv2.imshow("Canny Edges", cannyEdges)
    #cv2.imshow("Gray image", img)
    #cv2.imshow("Adaptive threshold image", adaptive_img)
    cv2.imshow("Local Otsu Thresholding", local_bin_img)
    cv2.imshow("Global Otsu Thresholding", global_bin_img)



    # define range of white color in HSV
    # change it according to your need !
    sensitivity = 180
    lower_white = np.array([255-sensitivity,255-sensitivity,255-sensitivity])
    upper_white = np.array([255,255,255])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv2, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(hsv2,hsv2, mask= mask)

    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    cv2.imshow('res2',hsv2)




    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    while(1):
        cannyMinVal = cv2.getTrackbarPos("cannyMin\n1", "Canny Edges")
        cannyMaxVal = cv2.getTrackbarPos("cannyMax\n1", "Canny Edges")
        sizefactor = cv2.getTrackbarPos("sizefactor\n1", "Canny Edges") / 100

        connect = cv2.connectedComponents(img)

        img = cv2.resize(global_bin_img, None, None, sizefactor,sizefactor, cv2.INTER_CUBIC)

        cannyEdges = cv2.Laplacian(img,cv2.CV_8U)
        #cannyEdges = cv2.Canny(img,cannyMinVal,cannyMaxVal)
        cv2.imshow("Canny Edges", cannyEdges)

        im2,contours,hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours
        M = cv2.moments(cnt[0])
        for c in cnt:
            x,y,w,h = cv2.boundingRect(c)
            epsilon = 0.05*cv2.arcLength(c,True)
            approx = len(cv2.approxPolyDP(c,epsilon,True))
            area = cv2.contourArea(c,True)
            #if approx ==4:
            cv2.rectangle(im2,(x,y),(x+w,y+h),(100,55,55),2)
                
            #print(area)
                

        cv2.imshow("im2", im2)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cv2.destroyAllWindows()



if __name__ == "__main__":
    main();

