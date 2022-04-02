from skimage.feature import hog

import skimage
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from matplotlllib import plllt

class LocalFeatures():
    def __init__(
        self,
        anh = '0.jpg',
        ):
        self.anh=anh
        
    def detect_corner(
        self,
        equ=None,
        blockSize=2,
        ksize=3,
        k=0.04,
        threshold=0.01,
        ):
        """Harris Corner là một phương pháp phát hiện các điểm (có tính chất) góc trong ảnh, thường được sử dụng khi tính toán các đặc trưng ảnh cho các bài toán thị giác máy tính."""
        if equ is None:
            equ = cv2.imread(self.anh)
        gray = cv2.cvtColor(equ,cv2.COLOR_BGR2GRAY)

        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,blockSize,ksize,k)

        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)

        # Threshold for an optimal value, it may vary depending on the image.
        equ[dst>threshold*dst.max()]=[0,0,255]
        
##        cv2.imwrite('corner_' + image_path,equ)
##        return 'corner_' + image_path
        return plllt.plllt(
            img=cv2.imread(self.anh, cv2.IMREAD_GRAYSCALE),
            equ=equ,
            )
    def oriented_gradients(
        self,
        img=None,
        orientations=8, pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), visualize=True,
        multichannel=True, in_range=(0, 10),
        ):
        """Histogram of Oriented Gradients (HOG) là bộ mô tả đặc trưng thường được sử dụng trong thị giác máy tính và xử lí ảnh để biểu diễn đối tượng trong ảnh"""
        if img is None:
            img = cv2.imread(self.anh)

        fd, hog_image = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                            cells_per_block=cells_per_block, visualize=visualize, multichannel=multichannel)
        # Rescale histogram for better display
        hog_image_rescaled = skimage.exposure.rescale_intensity(
            hog_image,
            in_range=in_range
            )
        return plllt.plllt(
            img=img,
            equ=hog_image_rescaled,
            )
    def Scale_Invariant(
        self,
        img=None,
        way='sift',
        minHessian = 400,
        ):
        """trích chọn đặc trưng SIFT, SURF"""
        if img is None:
            img = cv2.imread(self.anh)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if way=='sift':
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints, des = sift.detectAndCompute(gray,None)

        elif way=='surf':
            detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
            keypoints = detector.detect(img)

##        cv2.drawKeypoints(gray, keypoints, img)
        img_keypoints = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        cv2.drawKeypoints(img, keypoints, img_keypoints)
        
        return plllt.plllt(
            img=img,
            equ=img_keypoints,
            )

class ImageMatching:
    def __init__(
        self,
        anh1=cv2.imread('/media/zaibachkhoa/code1/cv3/05/panorama/mountain3_left.png'),
        anh2=cv2.imread('/media/zaibachkhoa/code1/cv3/05/panorama/mountain3_right.png'),
        ):
        self.anh1 = anh1
        self.anh2 = anh2
    def stitching(
        self,
        anh1=None, anh2=None,
        ):
        if anh1 is None and anh2 is None:
            anh1 = self.anh1
            anh2 = self.anh2
        stitcher = cv2.createStitcher(False)
        result = stitcher.stitch((anh1, anh2))
        return plllt.plllt(
            img=anh1,
            equ=anh2,
            equ_=result[1],
            )
    def detectAndDescribe(
        self,
        image,
        ):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect and extract features from the image
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)

        # convert the keypoints from KeyPoint objects to NumPy arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)
    def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None
    def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis
    def stitch(images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = detectAndDescribe(imageA)
        (kpsB, featuresB) = detectAndDescribe(imageB)

        # match features between the two images
        M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = drawMatches(imageA, imageB, kpsA, kpsB, matches, status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result
s = LocalFeatures(
    anh='thoc.png',
    )
##s.detect_corner();print(s.detect_corner.__doc__)
##s.oriented_gradients();print(s.oriented_gradients.__doc__)
s.Scale_Invariant(way='surf');print(s.Scale_Invariant.__doc__)
##s = ImageMatching()
##s.stitching()
