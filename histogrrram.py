import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlllib import plllt

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
        return plllt.plllt(img=img, equ=equ)
    def equalizeHistEachParrrt(self):
        img = cv2.imread(self.anh, cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equ = clahe.apply(img)
        return plllt.plllt(img=img, equ=equ)
    def changeByEachPixcel(self,
                           way='SquareTransfrom',
                           gamma=5,#0.04-25
                           ):
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
                elif way=='Gamma_Correction':
                    print(
                        """Với γ<1, các cùng ảnh ban đầu bị tối sẽ được tăng sáng và
histogram sẽ có xu hướng dịch chuyển sang phải,
ngược lại với γ>1, ảnh sẽ được giảm sáng."""
                        )
                    equ[i][j] = ((int(img[i][j])/255.0) ** gamma)*255
                else:
                    print('Lỗi: way=="SquareTransfrom"'+\
                          ' hoặc way=="SquareRootTransfrom"'+\
                          ' hoặc way=="Gamma_Correction"')
                    return None
        return plllt.plllt(img=img, equ=equ)
    def Filttter(
        self,
        way='AveragingKernel',
        ):
        img = cv2.imread(self.anh, cv2.IMREAD_COLOR)
        img = plllt.convertcolor(img)
        if way=='Bluuur':
            equ = cv2.blur(img, (7, 7), 0)
            equ_ = cv2.GaussianBlur(img, (9, 9), 0)#None 
            equ__ = cv2.medianBlur(img, 9)#None
            return plllt.plllt(
                img=img,
                equ=equ,
                equ_=equ_,
                equ__=equ__,
                )
        elif way=='ConvertScaleAbs':
            print("""𝑔(𝑖,𝑗)=𝛼⋅𝑓(𝑖,𝑗)+𝛽""")
            equ = cv2.convertScaleAbs(img, 1.1, 5)
            return plllt.plllt(img=img, equ=equ)
        elif way=='AveragingKernel':
            kernel = np.ones((5,5),np.float32)/25
        elif way=='SharpeningKernel':
            kernel = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
        else:
            print('Lỗi: way=="Bluuur"'+\
                  ' hoặc way=="AveragingKernel"'+\
                  ' hoặc way=="ConvertScaleAbs"'+\
                  ' hoặc way=="SharpeningKernel"')
            return None
        equ = cv2.filter2D(img, -1, kernel)
        return plllt.plllt(img=img, equ=equ)

s = XulyanhBangHistogram()
##s.changeByEachPixcel(way='SquareTransfrom')
##s.changeByEachPixcel(way='SquareRootTransfrom')
##s.changeByEachPixcel(way='Gamma_Correction')
##s.equalizeHiiist()
##s.equalizeHistEachParrrt()
s.Filttter(way='ConvertScaleAbs')
