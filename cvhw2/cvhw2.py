import cv2
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk
import numpy as np

def onCannyParamChange(x):
    pass

def main():
    img = cv2.imread("twoimage.jpg")
    img_o = img.copy()
    img = cv2.resize(img, None, None, 1, 1, cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.equalizeHist(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #img = clahe.apply(img)
    kernel = np.ones((3,3), np.uint8)

    #img = cv2.medianBlur(img, 3)
    r = 70
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
    local_bin_img = local_bin_img.astype(np.uint8)
    local_bin_img = cv2.dilate(local_bin_img, kernel)
    local_bin_img = cv2.morphologyEx(local_bin_img, cv2.MORPH_OPEN, kernel)
    local_bin_img = cv2.medianBlur(local_bin_img, 11)
    local_bin_img = cv2.medianBlur(local_bin_img, 3)

    #cv2.imshow("Original Image", img_o)
    #cv2.imshow("Canny Edges", cannyEdges)
    #cv2.imshow("Gray image", img)
    #cv2.imshow("Adaptive threshold image", adaptive_img)
    #cv2.imshow("Local Otsu Thresholding", local_bin_img)
    #cv2.imshow("Global Otsu Thresholding", global_bin_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    while(1):
        cannyMinVal = cv2.getTrackbarPos("cannyMin\n1", "Canny Edges")
        cannyMaxVal = cv2.getTrackbarPos("cannyMax\n1", "Canny Edges")
        sizefactor = cv2.getTrackbarPos("sizefactor\n1", "Canny Edges") / 100

        img = cv2.resize(local_bin_img, None, None, sizefactor,sizefactor, cv2.INTER_CUBIC)

        cannyEdges = cv2.Canny(img,cannyMinVal,cannyMaxVal)
        sobel = cv2.Sobel(local_bin_img, cv2.CV_64F,1,1,ksize=5)

        cv2.imshow("sobel", sobel)
        cv2.imshow("Canny Edges", cannyEdges)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cv2.destroyAllWindows()



if __name__ == "__main__":
    main();

