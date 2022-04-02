import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlllib import plllt

class Detect_edge():
    def __init__(
        self,
        anh='thoc.png',
        ):
        self.anh = anh

    
    def edge_detection(
        self,
        way='Sobel_basic',
        image_path=None,
        blur_ksize=5,
        sobel_ksize=1,
        skipping_threshold=30,
        ):
        if image_path==None:
            image_path = self.anh
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gaussian = cv2.GaussianBlur(gray,(blur_ksize,blur_ksize),0)

        if way=='Sobel_basic':#sobel
            img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=sobel_ksize)
            img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=sobel_ksize)

            img_ = (img_sobelx + img_sobely)/2
        elif way=='Sobel':
            sobelx64f = cv2.Sobel(img_gaussian,cv2.CV_64F,1,0,ksize=sobel_ksize)
            abs_sobel64f = np.absolute(sobelx64f)
            img_sobelx = np.uint8(abs_sobel64f)

            sobely64f = cv2.Sobel(img_gaussian,cv2.CV_64F,1,0,ksize=sobel_ksize)
            abs_sobel64f = np.absolute(sobely64f)
            img_sobely = np.uint8(abs_sobel64f)

            img_ = (img_sobelx + img_sobely)/2
        elif way=='prewitt':#prewitt
            kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
            img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
            img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
            img_prewitt1 = (img_prewittx + img_prewitty)/2
            
            kernelx2 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
            kernely2 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
            img_prewittx2 = cv2.filter2D(img_gaussian, -1, kernelx2)
            img_prewitty2 = cv2.filter2D(img_gaussian, -1, kernely2)
            img_prewitt2 = (img_prewittx2 + img_prewitty2)/2
            
            img_ = (img_prewitt1 + img_prewitt2)/2
        elif way=='canny':
            img_ = cv2.Canny(
                img_gaussian,
                threshold1=100,
                threshold2=200,
                )
        else:
            print('Lỗi: way=="Sobel_basic"'+\
                  ' hoặc way=="Sobel"'+\
                  ' hoặc way=="prewitt"'+\
                  ' hoặc way=="canny"')
            return None
            
        for i in range(img_.shape[0]):
            for j in range(img_.shape[1]):
                if img_[i][j] < skipping_threshold:
                    img_[i][j] = 0
                else:
                    img_[i][j] = 255
        return img_

test = Detect_edge()
equ=test.edge_detection(
    way='canny',
    skipping_threshold=10,
    )
if equ is not None:
    plllt.plllt(
        img=cv2.imread(test.anh, cv2.IMREAD_GRAYSCALE),
        equ=equ,
        )
