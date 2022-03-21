import cv2
import numpy as np
import matplotlib.pyplot as plt

class XulyanhBangHistogram():
    def __init__(
        self,
        anh='0.jpg',
        ):
        self.anh = anh 
        
    def histog(self, show=False):
        img = cv2.imread(self.anh)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(gray.shape)
        histogram = np.zeros((256, ))
        print(histogram.shape)
        
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                histogram[gray[i][j]] += 1
                
        if show==True:
            plt.plot(histogram);plt.show()
        return histogram
    def histog_(self):
        img = cv2.imread(self.anh, cv2.IMREAD_GRAYSCALE)
        plt.hist(img.ravel(),256,[0,256])
        return plt.show()
    def histog__(self):
        img = cv2.imread(self.anh, cv2.IMREAD_COLOR)
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv2.calcHist([img],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
        return plt.show()
    def equalizeHiiist(self):
        img = cv2.imread(self.anh, cv2.IMREAD_GRAYSCALE)
        equ = cv2.equalizeHist(img)
        return self.plllt(img, equ)
    def equalizeHistEachParrrt(self):
        img = cv2.imread(self.anh, cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equ = clahe.apply(img)
        return self.plllt(img, equ)
    def equalizeHistEachPixcel(self, way='SquareTransfrom'):
        img = cv2.imread(self.anh, cv2.IMREAD_GRAYSCALE)
        equ = img.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
##                equ[i][j] = int(np.square(int(equ[i][j]))/255)\
##                            if way=='SquareTransfrom' else\
##                            int(np.sqrt(int(equ[i][j])))#SquareRootTransfrom
                if way=='SquareTransfrom':
                    equ[i][j] = int(np.square(int(equ[i][j]))/255)
                elif way=='SquareRootTransfrom':
                    equ[i][j] = int(np.sqrt(int(equ[i][j])))
                else:
                    print('Lỗi: "SquareTransfrom" hoặc "SquareRootTransfrom"')
                    return None
        return self.plllt(img, equ)
    def plllt(self, img, equ):
        plt.subplot(2,2,1)
        plt.imshow(img, cmap='gray')
        plt.subplot(2,2,2)
        plt.hist(img.ravel(),256,[0,256])
        plt.subplot(2,2,3)
        plt.imshow(equ, cmap='gray')
        plt.subplot(2,2,4)
        plt.hist(equ.ravel(),256,[0,256])
        plt.show()

s = XulyanhBangHistogram()
##s.equalizeHistEachPixcel(way='SquareTransfrom')
s.equalizeHistEachPixcel(way='SquareRootTransfrom')
