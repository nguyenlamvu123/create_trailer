import cv2
import matplotlib.pyplot as plt

class Matplll():
    def __init__(
        self,
        anh='0.jpg',
        ):
        self.anh = anh
        
    def histog_(
        self,
        anh=None,
        ):
        if anh==None:
            anh=self.anh,#None
        img = cv2.imread(anh, cv2.IMREAD_GRAYSCALE)
        plt.hist(img.ravel(),256,[0,256]);plt.show()

    def plllt(
        self,
        img=None,
        equ=None,
        ):
        if img is None and equ is None:
            img = cv2.imread(self.anh, cv2.IMREAD_GRAYSCALE)
            equ = img.copy()
        plt.subplot(2,2,1)
        plt.imshow(img, cmap='gray')
        plt.subplot(2,2,2)
        plt.hist(img.ravel(),256,[0,256])
        plt.subplot(2,2,3)
        plt.imshow(equ, cmap='gray')
        plt.subplot(2,2,4)
        plt.hist(equ.ravel(),256,[0,256])
        plt.show()

##def plllt(anh=None):
##    if anh:
##        return Matplll(anh=anh)
##    return Matplll()
plllt = Matplll()
