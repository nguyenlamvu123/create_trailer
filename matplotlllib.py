import cv2
import matplotlib.pyplot as plt

def histog_(
    anh,
    ):
    img = cv2.imread(anh, cv2.IMREAD_GRAYSCALE)
    plt.hist(img.ravel(),256,[0,256]);plt.show()

def subpl(
    anh,
    ):
    img = cv2.imread(anh, cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img)
    
    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.subplot(2,2,2)
    plt.hist(img.ravel(),256,[0,256])
    plt.subplot(2,2,3)
    plt.imshow(equ, cmap='gray')
    plt.subplot(2,2,4)
    plt.hist(equ.ravel(),256,[0,256])
    plt.show()

