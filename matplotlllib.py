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
        equ_=None,
        equ__=None,
        ):
        #plt.title('Sobel Image'),plt.xticks([]),plt.yticks([])
        hang = 2
        if img is None and equ is None:
            img = cv2.imread(self.anh, cv2.IMREAD_GRAYSCALE)
            equ = img.copy()
        if equ_ is not None:
            hang = 3
            if equ__ is not None:
                hang = 4
                plt.subplot(hang,2,7)#.axis('off')
                plt.imshow(equ__, cmap='gray')
                plt.subplot(hang,2,8)#.axis('off')
                plt.hist(equ__.ravel(),256,[0,256])
            plt.subplot(hang,2,5)#.axis('off')
            plt.imshow(equ_, cmap='gray')
            plt.subplot(hang,2,6)#.axis('off')
            plt.hist(equ_.ravel(),256,[0,256])
        plt.subplot(hang,2,1)#.axis('off')
        plt.imshow(img, cmap='gray')
        plt.subplot(hang,2,2)#.axis('off')
        plt.hist(img.ravel(),256,[0,256])
        plt.subplot(hang,2,3)#.axis('off')
        plt.imshow(equ, cmap='gray')
        plt.subplot(hang,2,4)#.axis('off')
        plt.hist(equ.ravel(),256,[0,256])
        plt.show()
##        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
##
##        ax1.axis('off')
##        ax1.imshow(img, cmap=plt.cm.gray)
##        ax1.set_title('Input image')
##        ax2.axis('off')
##        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
##        ax2.set_title('Histogram of Oriented Gradients')
        
##        plt.show()

    def convertcolor(
        self,
        img=None,
        ):
        if img is None:
            img = cv2.imread(self.anh)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

##def plllt(anh=None):
##    if anh:
##        return Matplll(anh=anh)
##    return Matplll()
plllt = Matplll()
