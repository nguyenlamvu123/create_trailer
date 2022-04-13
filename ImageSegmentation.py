from sklearn.cluster import MeanShift

import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import tensorflow as tf
import cv2
import os.path
import warnings
from distutils.version import LooseVersion
import glob

##import helper
##import project_tests as tests

class ImageeeSegmentation:
    def __init__(
        self,
        anh='0.jpg',#'thoc.png',#
        ):
        self.anh = cv2.imread(anh)
    def MeanShift(
        self,
        reshap=3,
        X_=None,
        ):
        """Thuật toán Mean Shift cho kết quả là các Superpixel có sự tương đồng về màu sắc và vị trí gần nhau trong không gian"""
        def init_seed(X=None, k=None):
            """Đối với MeanShift, ta cần một hàm để khởi tạo ngẫu nhiên các hạt seed"""
            return X[np.random.choice(X.shape[0], k, replace=False)]
        img = self.anh#;plt.imshow(img, cmap='gray');plt.show()
        if reshap==1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X = X_ if X_ is not None\
            else img.reshape((-1, reshap))
        random_seeds = init_seed(X, reshap*10)

        ms = MeanShift(
            bandwidth=2,
            seeds=random_seeds,
            );print('ms: ', ms)
        
        import time#;print('*')
        start_time = time.time()#;print('**')
        ms.fit(X)
        seconds = time.time() - start_time#;print('***')
        print('ms.fit(X) taken:', time.strftime("%H:%M:%S",time.gmtime(seconds)))
                
        center = ms.cluster_centers_#;print('****')
        #print(center)
        label = ms.labels_#;print('*****')
        #print(label.shape)
        
        segmented_image = center[label]#;print('******')
        if reshap==3: segmented_image = segmented_image[:, :3]
        segmented_image = np.reshape(segmented_image, img.shape)#;print('*******')
        # plt.subplots(1, 1, figsize=(12,9))
        if reshap==1:
            plt.imshow(segmented_image/255.0, cmap='gray')
        else:
            plt.imshow(segmented_image/255.0)
        plt.show()
    def MeanShift_grayimg(self):
        """biểu diễn các mỗi điểm ảnh bằng một vecto 1 chiều (không gian đặc trưng 1D): giá trị độ sáng tại điểm ảnh tương ứng"""
        self.MeanShift(
            reshap=1,
            )
    def MeanShift_rgb3d(self):
        """mỗi điểm ảnh sẽ được biểu diễn bởi một vector 3 chiều tương ứng là giá trị màu R, G, B tại điểm ảnh tương ứng"""
        self.MeanShift()
    def MeanShift_rgb5d(self):
        """mỗi điểm ảnh sẽ được biểu diễn bởi một vector 5 chiều có dạng (R, G, B, x, y), trong đó (R, G, B) là cường độ màu tại điểm ảnh và (x, y) là tọa độ điểm ảnh đó"""
        def get_5D_vector(img):
            X = None
            
            X_pos = np.zeros((img.shape[0], img.shape[1], 2))
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    X_pos[i][j][0] = i
                    X_pos[i][j][1] = j
            X = img.reshape((-1, 3))
            X_pos = X_pos.reshape((-1, 2))

            X = np.concatenate((X, X_pos), axis=1)
            return X
        self.MeanShift(
##            X_=get_5D_vector(self.anh),
            X_=get_5D_vector(
                cv2.GaussianBlur(
                    self.anh,
                    (9,9),
                    0
                    )
                ),
            )

##print(init_seed.__doc__)
s=ImageeeSegmentation()
##s.MeanShift_grayimg()
##s.MeanShift_rgb3d()
s.MeanShift_rgb5d()
