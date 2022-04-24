import cv2
import numpy as np

class MotionEstimObjTracking:
    def __init__(
        self,
        possstGMM=True,
        video_path='./video/Flair.mp4',
        ):
        self.video_path = video_path
        self.vs = cv2.VideoCapture(
            self.video_path
            )# Initialize video reader object
        cv2.namedWindow("Video")
        self.possstGMM = possstGMM
        self.kalman = self.create_kalman()
        self.list_detection = []# List detection by background subtraction
        self.list_predict = []# List estimation by kalman filter
        framerate = self.vs.get(5)#s.vs.get(cv2.CAP_PROP_FPS)
        print("framerate: ", framerate)
        framecount = self.vs.get(7)#s.vs.get(cv2.CAP_PROP_FRAME_COUNT)
        print("framecount: ", framecount)
        cv2.createTrackbar("_",
                           "Video",
                           0, int(framecount),
                           getFrame)
        cv2.createTrackbar("Speed",
                           "Video",
                           25, 100,
                           setSpeed)

    def MotionEstimate(
        self,
        frame,
        ):
        """phát hiện chuyển động với mô hình trừ nền Gaussian Mixture Model được cài đặt sẵn trong thư viện OpenCV"""
        fgbg = cv2.createBackgroundSubtractorMOG2()# Initialize GMM object
        return fgbg.apply(frame)# Apply GMM model to frame
    def create_kalman(self):
        dt = 0.2
        kalman = cv2.KalmanFilter(4, 2, 0)
        kalman.transitionMatrix = np.array(
            [[1., 0., dt, 0.], [0., 1., 0., dt], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=np.float32)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kalman.processNoiseCov = 0.5 * np.array([[dt ** 4.0 / 4.0, 0., dt ** 3.0 / 2.0, 0.],
                                                 [0., dt ** 4.0 / 4.0, 0., dt ** 3.0 / 2.0],
                                                 [dt ** 3.0 / 2.0, 0., dt ** 2.0, 0.],
                                                 [0., dt ** 3.0 / 2.0, 0., dt ** 2.0]], dtype=np.float32)
        kalman.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)
        kalman.errorCovPost = 1e-1 * np.eye(4, dtype=np.float32)
        return kalman

    def postGMM(self, thresh, frame):
        thresh = cv2.medianBlur(thresh, 5)# Remove noise
        thresh = cv2.dilate(thresh, None, iterations=2)# Dilate
        #sử dụng phép biến đổi dilate để nối các block cùng màu lại với nhau
        _, thresh = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY)
        # Threshold-> convert to binary image        
        _, cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find contours with maximum area
        maxArea = 0
        maxContour = None
        for c in cnts:
            if (cv2.contourArea(c) > maxArea) and (cv2.contourArea(c) >= 500):
                maxArea = cv2.contourArea(c)
                maxContour = c

        if maxContour is None:
            self.list_detection = []
            self.list_predict = []
        else:
            # Draw result
            (x, y, w, h) = cv2.boundingRect(maxContour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            dx = int(x + w / 2)
            dy = int(y + h / 2)
            self.list_detection.append((dx, dy))

            if len(self.list_detection) == 1:
                self.kalman = self.create_kalman()
                self.kalman.statePre = np.array([[dx], [dy], [0.], [0.]], dtype=np.float32)
            # Kalman correct
            self.kalman.correct(np.array([[dx], [dy]], dtype=np.float32))
            # Kalman predict
            estimate = self.kalman.predict()
            self.list_predict.append((estimate[0, 0], estimate[1, 0]))

            if len(self.list_detection) > 1:
                for i in range(len(self.list_detection) - 1):
                    x, y = self.list_detection[i]
                    u, v = self.list_detection[i + 1]
                    cv2.line(frame, (x, y), (u, v), (0, 0, 255))

                # Draw tracker
                for i in range(len(self.list_detection) - 1):
                    x, y = self.list_predict[i]
                    u, v = self.list_predict[i + 1]
                    cv2.line(frame, (x, y), (u, v), (255, 0, 0))
        return thresh

    def act(self):
        _, frame = self.vs.read()# Read video

        # If end of video, break
        if frame is None:
            return False#break

        # resize the frame, easy to display
        ##import imutils
        ##frame = imutils.resize(frame, width=500)
        f = 500/frame.shape[1]
        frame = cv2.resize(
            frame,
            (0,0),
            fx=f,#500/400,
            fy=f,#500/400,
            )
########################################################################################################
##        frameWidth = int(self.vs.get(cv2.CAP_PROP_FRAME_WIDTH))
##        frameHeight = int(self.vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
##        cv2.putText(img=frame,
##                    text='hahahehehoho',
##                    org=(int(frameWidth/2), int(frameHeight/2)),
##                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
##                    fontScale=30,
##                    color=(0, 255, 0))
########################################################################################################    
        thresh = self.MotionEstimate(frame)# Apply GMM model to frame
        if self.possstGMM:
            thresh = self.postGMM(thresh, frame)

        # Display result
        cv2.imshow("Video", frame)
        cv2.imshow("GMM", thresh)
        return True 

def getFrame(frame_nr):
    global s
    s.vs.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
def setSpeed(val):
    global playSpeed
    playSpeed = max(val,1)

playSpeed = 10
s = MotionEstimObjTracking(
    possstGMM=False,
    video_path='./video/uavdemo_sta.avi',
    )
##s.vs.set(cv2.cv2.CAP_PROP_POS_FRAMES, 3000)
while True:
    keey = s.act()#;print(keey)
########################################################################################################    
####    cv2.setTrackbarPos(
####        "_",
####        "Video",
####        int(s.vs.get(cv2.CAP_PROP_POS_FRAMES))
####        )
##    getvalue = s.vs.get(0)#;print('###', getvalue)
##    if 20000<getvalue<21000: print('hahahehehoho')
########################################################################################################    
    key = cv2.waitKey(playSpeed) & 0xFF
    if key == ord("q") or key == 27 or not keey:
        break#return False#
##vs = cv2.VideoCapture('./video/Flair.mp4')
##while True:
##    _, frame = vs.read()
##    if frame is None:
##        break
##    cv2.imshow("Original", frame)
##    key = cv2.waitKey(10) & 0xFF
##    if key == ord("q"):
##        break
cv2.destroyAllWindows()
