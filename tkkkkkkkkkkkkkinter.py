#!/usr/bin/python
# -*- coding: utf8 -*-
####gpu:
##h264_amf (H.264), hevc_amf (H.265 or HEVC) to access AMD gpu, (windows only)
##h264_nvenc use nvidia gpu cards (work with windows and linux)
####
##ffmpeg -hwaccel auto -i input output# -hwaccel cuda # -hwaccel_device 0 
##ffmpeg -c:v h264_cuvid -i input output
####
##ffmpeg -hwaccel auto -hide_banner -vsync 1 -i file.mp4 -c:v h264_amf -c:a copy -ss 10 -t 100 -gpu all output.mp4

##02.01.2021: thử video 5 phút, đoạn 1/3 video đầu tiên ngắt thành 10 đoạn
##thì 9 đoạn có thời lượng nhỏ hơn giới hạn dưới (line 845) nên
##list diiiiiiiiemtrichdoan (line 843) chỉ có 1 phần tử
##soluong=3 nên k bị tràn (line 857)
##thử import kmeans của h2o4gpu để dùng gpu (line 80)

from tkinter import filedialog, Grid
import tkinter as tk
from shutil import copy2
import time, os


root = tk.Tk()
##root.withdraw()

root.title("Phần mềm trích xuất phim")
root.geometry('1000x400')

root.source = filedialog.askopenfilename(initialdir = "/",title = "Select source file",filetypes = (("video files",".mp4"),("all files",".*")))
Grid.rowconfigure(root, 0)#, weight=1)
Grid.columnconfigure(root, 0, weight=1)

greeting = label = tk.Label(
    text="Move file\n" + root.source,
    foreground="green",  # Set the text color to white#fg="white"
    background="black",  # Set the background color to black#bg="black"
    width=10,
    height=10#20
)
##greeting.pack(fill=tk.X, side=tk.LEFT, expand=True)#tk.X, tk.Y, tk.BOTH 
greeting.grid(row=0,column=0,sticky='EWNS')
##
##print ("Selected file " + root.source)
time.sleep(1)
root.target = os.path.join(os.getcwd(), 'videooooooooooooo', 'videooooooooooooo')#r'.\videooooooooooooo\videooooooooooooo'
##root.target = filedialog.askdirectory(initialdir = "/",title = "Select target directory")

grting = label = tk.Label(
    text="To directory\n" + root.target,
    foreground="red",  # Set the text color to white#fg="white"
    background="black",  # Set the background color to black#bg="black"
    width=10,
    height=10#20
)
##grting.pack(fill=tk.X, side=tk.LEFT, expand=True)#tk.X, tk.Y, tk.BOTH 
grting.grid(row=2,column=0,sticky='EWNS')

for zszs in os.listdir(root.target):
    if zszs[-4:]=='.mp4':
        os.remove(os.path.join(root.target, zszs))
copy2(root.source, root.target, follow_symlinks=True)
for zszs in os.listdir(root.target):
    if zszs[-4:]=='.mp4':
        os.rename(os.path.join(root.target, zszs), os.path.join(root.target, 'file.mp4'))
        break

import warnings
warnings.filterwarnings("ignore")

import pickle, cv2, csv, datetime, json, sklearn, librosa
from pydub import AudioSegment,silence
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from pydub import AudioSegment,silence
from scipy.io import wavfile
from multiprocessing import Pool

##from sklearn.cluster import KMeans
from h2o4gpu.solvers import KMeans
from scipy.spatial.distance import cdist

soluong=3#5
def thaaaaaaaamsssssssssso():
    min_silence_len=1500#1700#2000#2100#
    giiiiiiiiiiioihanthoigian=10#20#40#giới hạn thời gian tối thiểu mỗi trích đoạn, đơn vị giây
    gioihantren=20#50#80#60*2#60*15#giới hạn thời gian tối đa mỗi trích đoạn, đơn vị giây
    silence_thresh=-23#-16#-18#-32#-22#-28#-20#-26#
    filllllllm = os.path.join(os.getcwd(), 'videooooooooooooo', 'videooooooooooooo', 'file.mp4')
    len___silence=6#3#8#
    return min_silence_len, giiiiiiiiiiioihanthoigian, gioihantren, silence_thresh, filllllllm, len___silence

def myFunc(e):
        return e[0]

min_silence_len, giiiiiiiiiiioihanthoigian, gioihantren, silence_thresh, filllllllm, len___silence = thaaaaaaaamsssssssssso()
def gioihanthoigiannnnnnnn(silence, gioihanduoi=30, gioihantren=60*15):
    sileeeeeeeeeeeeeeeeeeence = []
    k=0#chỉ số phần tử chạy từ đầu đến cuối tập silence
    samxinhdep=len(silence)
    while k<=samxinhdep-1:
            silenceeeeeeeeeeeeeeeee = silence[k]
            thoooooooigian = silenceeeeeeeeeeeeeeeee[1]-silenceeeeeeeeeeeeeeeee[0]
            if thoooooooigian < gioihanduoi:#nếu không đủ thời gian giới hạn
                print(silenceeeeeeeeeeeeeeeee[1],'-',silenceeeeeeeeeeeeeeeee[0],'<',gioihanduoi)
                if k==samxinhdep-1:
                    sileeeeeeeeeeeeeeeeeeence.append(silence[k])
                else:
                    silence[k+1]=(silence[k][0],silence[k+1][1])
            elif thoooooooigian > gioihantren:
                print(silenceeeeeeeeeeeeeeeee[1],'-',silenceeeeeeeeeeeeeeeee[0],'>',gioihantren)
                kk=0#1
                while kk <= thoooooooigian//gioihantren:
                    if kk!= thoooooooigian//gioihantren:
                        sammmmmmmmmmmmmmmxinhdep=(silence[k][0]+(kk)*gioihantren,silence[k][0]+(kk+1)*gioihantren)
                        sileeeeeeeeeeeeeeeeeeence.append(sammmmmmmmmmmmmmmxinhdep)
                    else:
                        if silence[k][1]-(silence[k][0]+kk*gioihantren)>=gioihanduoi:
                            sammmmmmmmmmmmmmmxinhdep=(silence[k][0]+kk*gioihantren,silence[k][1])
                            sileeeeeeeeeeeeeeeeeeence.append(sammmmmmmmmmmmmmmxinhdep)
##                    sileeeeeeeeeeeeeeeeeeence.insert(kk+k, silence[k+kk-1])
                    print(k, '____________', sammmmmmmmmmmmmmmxinhdep)
                    kk+=1
##                x, sr = rrrreaddata(fiiiiiiiiiiiiiile = filllllllm,
##                                    start = silenceeeeeeeeeeeeeeeee[0],
##                                    stop = silenceeeeeeeeeeeeeeeee[1])
##                spectralccccccccentroids(x, sr)
##                plt.title(str(silenceeeeeeeeeeeeeeeee[0])+'-'+str(silenceeeeeeeeeeeeeeeee[1])+'.png')
##                plt.savefig('zz'+str(silenceeeeeeeeeeeeeeeee[0])+'-'+str(silenceeeeeeeeeeeeeeeee[1])+'.png')
##                plt.show()
            else:
                print(k, '____________', silence[k])
                sileeeeeeeeeeeeeeeeeeence.append(silence[k])
            k+=1
    print(sileeeeeeeeeeeeeeeeeeence)
    print(len(sileeeeeeeeeeeeeeeeeeence))
    return sileeeeeeeeeeeeeeeeeeence

def create_features_bow(image_descriptors, BoW, num_clusters):
    X_features = []

    for i in range(len(image_descriptors)):
        features = np.array([0] * num_clusters)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)#tính khoảng cách
            argmin = np.argmin(distance, axis=1)#xem vị trí tâm cụm gần
                                                # điểm đó nhất ở đâu

            for j in argmin:
                features[j] += 1

        X_features.append(features)

    return X_features

def load_image(image_path):
    return cv2.imread(image_path)

def statistic():
    label = []
    num_images = []
    
    for lab in os.listdir('trainingset'):
        label.append(lab)
        num_images.append(len(os.listdir(os.path.join('trainingset', lab))))

    return label, num_images

def read_data(label2id, folder = 'trainingset_audio'):
    X = []
    Y = []
    
    for label in os.listdir(folder):
        for img_file in os.listdir(os.path.join(folder, label)):
            #### YOUR CODE HERE ####
            if str(img_file)[-4:]=='.png':
                img = load_image(os.path.join(folder, label, img_file))
##                cv2.imshow('', img)
##                cv2.waitKey()
                if img is None:
                    print(str(os.path.join(folder, label, img_file))+'________None_ln58_________', img)
            #### END YOUR CODE #####
                X.append(img)
                Y.append(label2id[label])
    
    return X, Y

def rrrreaddata(fiiiiiiiiiiiiiile = 'old-car-engine_daniel_simion.wav', start = None, stop = None):
    if start and stop:
        x , sr = librosa.load(fiiiiiiiiiiiiiile, offset=start, duration=stop-start)#)#
    else:
        x , sr = librosa.load(fiiiiiiiiiiiiiile)#, offset=start, duration=stop-start)#
    plt.figure(figsize=(12, 4))# Make a new figure
    return x, sr

# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

def timmmmmmmmmme(feeeeea):
    # Computing the time variable for visualization
    frames = range(len(feeeeea))
    t = librosa.frames_to_time(frames)
    return t

def detectttttttttttsilence(filllllllm = os.path.join(os.getcwd(), 'videooooooooooooo', 'videooooooooooooo', 'file.mp4'),
                            min_silence_len=2000,
                            silence_thresh=-22,
                            start=None,
                            stop=None,
                            len___silence=3):
    if start and stop:
            myaudio = AudioSegment.from_file(filllllllm)[start:stop]
    else:
            myaudio = AudioSegment.from_mp3(filllllllm)            
    print('load file success, silence deeeeeetecttttttttt...tt.tt')
    silllence, i=[], 0
    while len(silllence)<len___silence:
        i+=1
        from pydub import silence
        silllence = silence.detect_silence(myaudio, min_silence_len, silence_thresh)
        ##silllence = librosa.effects.split(myaudio, top_db=20)
        silllence = [(round(start/1000),round(stop/1000)) for start,stop in silllence] #convert to sec
        print('tham số min_silence_len là ', min_silence_len, ',tham số silence_thresh là ', silence_thresh, ', chia thành ', len(silllence), ' đoạn, yêu cầu không dưới ', len___silence)
        min_silence_len+=100
        silence_thresh-=6#2
    
    print('chia thành ', len(silllence), ' đoạn')
    return silllence, i, min_silence_len-100, silence_thresh+6

def spectralccccccccentroids(x, sr):
    spectral_centroids = librosa.feature.spectral_centroid(x, sr)[0]
    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    t = timmmmmmmmmme(spectral_centroids)
    plt.plot(t, normalize(spectral_centroids), color='b')
##.spectral_centroid is used to calculate the spectral centroid for each frame.

def spectralrrrrrrrrrrolloff(x, sr):
    spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    t = timmmmmmmmmme(spectral_rolloff)
    plt.plot(t, normalize(spectral_rolloff), color='r')
##    Spectral rolloff is the frequency below which a specified percentage
##        of the total spectral dict

def mffcccccccc(x, sr):
    mfccs = librosa.feature.mfcc(x, sr=sr)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')

def chromaaaaaaaaaagram(x, sr, hop_length = 512):
    chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')

def display_mel_spectogram(x, sr, n_mels=128):    
    S = librosa.feature.melspectrogram(x, sr=sr, n_mels=n_mels)    
    log_S = librosa.power_to_db(S, ref=np.max)# Convert to log scale (dB)
    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')    
    plt.colorbar(format='%+02.0f dB')# draw a color bar
    plt.tight_layout()# Make the figure layout compact
                            
def Zeeeeeeerocrossingrate(x, sr):
    zcrs = librosa.feature.zero_crossing_rate(x + 0.0001)
    plt.plot(zcrs[0])

def extract_sift_features(X):
    image_descriptors = []
    sift = cv2.AKAZE_create()# tương tự phương pháp sift

    for i in range(len(X)):
        X[i] = cv2.normalize(X[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        kp, des = sift.detectAndCompute(X[i], None)
        if des is not None:
            #### YOUR CODE HERE ####
            image_descriptors.append(des)
            #### END YOUR CODE #####
        else:
            print(X[i])
            image_descriptors.append([np.zeros(61)])

    return image_descriptors

def kmeans_bow(all_descriptors, num_clusters):
    bow_dict = []

    kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)
    if kmeans:
        bow_dict = kmeans.cluster_centers_
    else:
        bow_dict = []
    return bow_dict

def labellllll2id(fillllllllllllle=os.path.join(os.getcwd(), 'trainingset_audio')):
    return {str(label): indexxxxxx for indexxxxxx, label in enumerate(os.listdir(fillllllllllllle))}
##    return {'tinh_cam':0, 'hanh_dong':1, 'hai_huoc':2, 'kinh_di':3, 'vo_thuat':4, 'chinh_kich':5, 'bi_an_giat_gan':6, 'phim_tai_lieu':7, 'coming-of-age':8}

def kkkkkkkkkkkkkkkkmmmeannnnnnn(X, Y, image_descriptors, BoW, img = r'6.Classification_KMean.jpg', i=0):#, start=None, stop=None):
    global num_clusters
    label2id = labellllll2id()
    if not os.path.isfile('zzmodel_create_features_bow.pkl'):
        if i ==0:    
            X_features = create_features_bow(image_descriptors, BoW, num_clusters)

            from sklearn.model_selection import train_test_split
            X_train = [] 
            X_test = []
            Y_train = []
            Y_test = []

            X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, random_state = 42)

            print('số phần tử tập train và kích thước của nó: ', len(X_train), X_train[0].shape)
            print('========================')
            print('số phần tử tập test và kích thước của nó: ', len(X_test), X_test[0].shape)

            from sklearn.svm import SVC
            
            from sklearn.model_selection import GridSearchCV
            parameter_candidates = [
              {'C': [0.001, 0.01, 0.1, 1, 5, 10, 100, 1000]},#, 'kernel': ['linear']},
            ]
            model = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)
            model.fit(X_train, Y_train)
            print('Best score:', model.best_score_)
            print('Best C:',model.best_estimator_.C)
            
            model = SVC(C = model.best_estimator_.C, probability=True)#SVC(C = 100)

            model.fit(X_train, Y_train)
            with open('zzmodel_create_features_bow.pkl', 'wb') as file:
                pickle.dump(model, file)

        # Tính độ chính xác trên tập dữ liệu huấn luyện:
            print('độ chính xác trên tập dữ liệu train', model.score(X_train, Y_train))

        # Tính độ chính xác trên tập dữ liệu test:
            print("độ chính xác trên tập dữ liệu test", model.score(X_test, Y_test))

    ##    cv2.imshow('', img)
    ##    cv2.waitKey()
        elif i > 0:
            with open('zzmodel_create_features_bow.pkl', 'rb') as file:
                model = pickle.load(file)
    else:
        with open('zzmodel_create_features_bow.pkl', 'rb') as file:
            model = pickle.load(file)
    my_X = [cv2.imread(img)]
        
    my_image_descriptors = None
    my_X_features = None

    my_image_descriptors = extract_sift_features(my_X)

    my_all_descriptors = []
    for mydescriptors in my_image_descriptors:
        if mydescriptors is not None:
            for des in mydescriptors:
                my_all_descriptors.append(des)
            
    my_BoW = kmeans_bow(my_all_descriptors, num_clusters)#, 4)#

    my_X_features = create_features_bow(my_image_descriptors, my_BoW, num_clusters)#4)#phát hiện đặc trưng của đối tượng

    y_pred = None

#### YOUR CODE HERE ####
    predicted_vector = model.predict(my_X_features)#dự đoán đặc trưng vừa phát hiện thuộc vào lớp nào trong mô hình đã huấn luyện
##    y_pred = model.predict(my_X_features)
##############################################################################
    prob=model.predict_proba(my_X_features)#điểm số của từng lớp trong mô hình đã huấn luyện
    prob_per_class_dictionary = dict(zip(model.classes_, prob[0]))
##    print('str(prob)', str(prob),'\nstr(model.classes_)',str(model.classes_),'\nstr(prob_per_class_dictionary)',str(prob_per_class_dictionary))
##https://stackoverflow.com/questions/47455156/why-the-predict-pr soba-function-of-sklearn-svm-svc-is-giving-probability-greater
##    predicted_vector = model.predict_classes(my_X_features)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit_transform(os.listdir(os.path.join(os.getcwd(), 'trainingset_audio')))#(np.array(label2id.keys()))#
    predicted_class = le.inverse_transform(predicted_vector) #chuyển đổi từ số đã mã hóa của lớp thành tên lớp
##    print("The predicted class is:", predicted_class[0], '\n')
##    for sszzzzzzz in prob_per_class_dictionary.keys():
##        if prob_per_class_dictionary[sszzzzzzz]==max(prob_per_class_dictionary.values()):
##            print('\nđiểm của đoạn trích ', le.inverse_transform(np.array([sszzzzzzz]))[0], ' : ', prob_per_class_dictionary[sszzzzzzz], '--------', start, '-', stop, '\n')

    predicted_proba_vector = model.predict_proba(my_X_features) #điểm số của từng lớp trong mô hình đã huấn luyện
    predicted_proba = predicted_proba_vector[0] #điểm số của từng lớp trong mô hình đã huấn luyện
    iiiiiiiiiiiiii=[]
    for i in range(len(predicted_proba)): 
            category = le.inverse_transform(np.array([i])) #chuyển đổi từ số đã mã hóa của lớp thành tên lớp
            iiiiiiiiiiiiii.append(str(category[0])+ " : "+str(format(predicted_proba[i], '.32f')))
##############################################################################
#### END YOUR CODE #####
# Get your label name using label2id variable (define above)
##    from sklearn.metrics import accuracy_score as sccc
    os.remove(img)
    for key, value in label2id.items():
        if value == predicted_vector[0]:#y_pred[0]:
##            print('Your prediction: ', key)
##            prrrr = [[1]]
##            print('điểm dự đoán', sccc(prrrr, y_pred))
            return key, str(iiiiiiiiiiiiii), max(prob_per_class_dictionary.values())#, prob_per_class_dictionary[sszzzzzzz], (start,stop))

def mussssssssssic(sssss='trainingset_audio', start=None, stop=None, label2id=None):
    import librosa
    import datetime
    import librosa.display
    import matplotlib.pyplot as plt
    from pydub import AudioSegment
    from scipy.io import wavfile
##    label22222222222id = {s:0 for s in label2id.keys()}
##    i = 0
    for label in os.listdir(sssss):
            if not os.path.isfile(os.path.join(sssss, label)):
                for img_file in os.listdir(os.path.join(sssss, label)):
##                    t = datetime.datetime.now()
                    if str(img_file)[-4:] in ('.wav', '.mp4'):
                        immmg = str(img_file)[:-4]+'from'+str(start)+'to'+str(stop)+'.png'
                        iiiiiiiiiiiiimg = os.path.join(sssss, label, immmg)                    
                        if not os.path.isfile(iiiiiiiiiiiiimg):
                            fiiiiiiiiiiiiiile = os.path.join(sssss, label, img_file)

                            x , sr = rrrreaddata(fiiiiiiiiiiiiiile, start, stop)
                            display_mel_spectogram(x, sr)

                            print(sssss+'______________'+str(img_file)[:-4]+'from'+str(start)+'to'+str(stop)+'.png')
                            plt.savefig(iiiiiiiiiiiiimg)
##                        print('mussssssssssic______________', str(datetime.datetime.now()-t))
                        if sssss==r'videooooooooooooo':# and str(img_file)[-4:] in ('.wav', '.mp4'):
##                            print(iiiiiiiiiiiiimg)
                            return iiiiiiiiiiiiimg

def svcsvc(min_silence_len, giiiiiiiiiiioihanthoigian, gioihantren, silence_thresh, filllllllm, len___silence, label2id, silence, fps=24, iiiiiiiiiiiiiiiiiiiiiiiiiiiiii=0):
    label22222222222id = {s:0 for s in label2id.keys()}
    i = 0
    silennnce=[]
    t = datetime.datetime.now()
    ##dieeeeeeeeeeemtrichdoan = []
    with open('zzzzzzzzzzprrrre.txt', 'a+', encoding='utf-8') as prrrre:
        for start, stop in silence:
    ##    for innnnnnnnnnnnnnnnnnnd, (start,stop) in enumerate(silence):
            if stop-start>=giiiiiiiiiiioihanthoigian:#áp dụng giới hạn dưới
                img = mussssssssssic(r'videooooooooooooo', start, stop, label2id=label2id)
                key, iiiiiiiiiiiiii, diemtrichdoan = kkkkkkkkkkkkkkkkmmmeannnnnnn(X, Y, image_descriptors, BoW, img = img, i = i)#(X, Y, image_descriptors, BoW, img = cv2.imread(img), i = i, start=start, stop=stop)
                label22222222222id[key] += 1
        ##            đếm số đoạn thuộc các thể loại
                keyyy = '\n'+str(img)+' mang sắc thái âm thanh của thể loại '+str(key)+'\n'
                print(keyyy)
                prrrre.write(iiiiiiiiiiiiii)
                prrrre.write(keyyy)
                silennnce.append((diemtrichdoan, (start, stop)))#silence[innnnnnnnnnnnnnnnnnnd]=(diemtrichdoan, (start, stop))
                i=1#i += 1
    ##        dieeeeeeeeeeemtrichdoan.append((diemtrichdoan, (start, stop)))
    ##            gắn các trích đoạn với số điểm của nó
    print(label22222222222id)
    silennnce.sort(reverse=True, key=myFunc)#dieeeeeeeeeeemtrichdoan.sort(reverse=True, key=myFunc)
    ##    sắp xếp các trích đoạn giảm dần theo số điểm của nó
##    silennnce=silennnce[:soluong]#lấy ra một số đoạn có số điểm cao nhất
##    print(silennnce)#(dieeeeeeeeeeemtrichdoan)
    silence=[silennnce[kkkkkkk][1] for kkkkkkk in range(soluong)]#silence=[]
##    for kkkkkkk in range(soluong):#(len(silennnce)):#
##    ##    if silence[kkkkkkk][1][1]-silence[kkkkkkk][1][0]>=giiiiiiiiiiioihanthoigian:
##            silence.append(silennnce[kkkkkkk][1])
    ##        đưa lại list về dạng ban đầu chỉ gồm các mốc thời gian
    silence.sort(key=myFunc)#sắp xếp lại list theo trình tự thời gian
##    print('#######################', silence)#(dieeeeeeeeeeemtrichdoan)

##    silence.append(silennnce[0][1])
    ############################################################################
    jsssssonn = str(filllllllm)
    jsssssoooooooooooon = [{'start':start, 'stop':stop} for start, stop in silence]

    jssssson = {'url_video': jsssssonn, 'parts': jsssssoooooooooooon}

    with open("zzzzzzzzsplit_movie.json", "a+") as fout:#with open("zzsplit_movie.json", "w") as fout:
        json.dump(jssssson, fout)
    print('ghi xong json')
    ##
    global samxxxxxxxinhdep

    for k in range(len(silence)):#(int(soluong)):
            sss=silence[k][0]#sss=silence[k][1][0]#dieeeeeeeeeeemtrichdoan[k][1][0]
            ssss=silence[k][1]
            if ssss-sss>gioihantren:
                ssss=sss+gioihantren#silence[k][1]#dieeeeeeeeeeemtrichdoan[k][1][1]
            print(sss, ssss)#(silence[k])#(dieeeeeeeeeeemtrichdoan[k])#
            samxxxxxxxinhdep.append((sss, 'zzzsvcsvcoutput'+str(sss)+'-'+str(ssss)+'.mp4'))
            cutting_theeeeeeeeeee_video(inputtttt = str(filllllllm),
                                        outputttttt = 'zzzsvcsvcoutput'+str(sss)+'-'+str(ssss)+'.mp4',
                                        start = str(sss),
                                        stop = str(ssss-sss),
                                        fps = fps)
##    samxxxxxxxinhdep.sort(key=myFunc)
##    print(samxxxxxxxinhdep)
##
##    with open('zzzsvcsvclisssssssst.txt', 'a+', encoding='utf-8') as fiiiiiiiiiiiiiiina:
##        for samxxxxxxxinhdepppp in samxxxxxxxinhdep:
##            fiiiiiiiiiiiiiiina.write("file '"+samxxxxxxxinhdepppp[1]+"'")
##            fiiiiiiiiiiiiiiina.write("\n")
    print('chương trình đã chạy trong ', datetime.datetime.now()-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzs)
##    global button6
##    button6.destroy()

##    val = max(label22222222222id.values())
##    #print(int(val))
##    for labbbbb in label22222222222id.items():
##    ##        print(labbbbb[0]) if int(labbbbb[1])==int(val) else print('haahahehehoho')
##            if int(labbbbb[1])==int(val):
##                theloai = labbbbb[0]
##    print('bộ phim được dự đoán thuộc thể loại ', theloai)
    
##    theloai='samxxxxxxxxxinhdep'
##    while not theloai in label22222222222id.keys():
##        theloai = input('nhập tên thể loại mong muốn trích xuất ')##sssssssssssssssssssss = input("gõ enter")

##    x , sr = rrrreaddata("zzzzzzzzzzzzzzzzzzzzzzzmergedfile.mp4")
##    display_mel_spectogram(x, sr)
##
##    plt.savefig(os.path.join(os.getcwd(), 'trainingset_audio', theloai, str(datetime.datetime.now()).replace(' ', '.').replace(':', '.')+'.png'))

def ssvc(min_silence_len, giiiiiiiiiiioihanthoigian, gioihantren, silence_thresh, filllllllm, len___silence, label2id, silence, fps=24, iiiiiiiiiiiiiiiiiiiiiiiiiiiiii=0):
    silence = gioihanthoigiannnnnnnn(silence=silence, gioihanduoi=giiiiiiiiiiioihanthoigian, gioihantren = gioihantren)

    print('giới hạn mỗi đoạn dài ', giiiiiiiiiiioihanthoigian, 'giây, tối đa ', gioihantren, 'giây, chia lại thành ', len(silence), ' đoạn')

    jsssssonn = str(filllllllm)
    jsssssoooooooooooon = [{'start':start, 'stop':stop} for start, stop in silence]
    ##jsssssoooooooooooon = {'part'+str(sothutuuuuuuu):{'start':start, 'stop':stop} for sothutuuuuuuu, (start, stop) in enumerate(silence)}

    jssssson = {'url_video': jsssssonn, 'parts': jsssssoooooooooooon}
    ##jssssson = json.dumps(jssssson)#, indent=4)

    with open("zzzzzzzzzzzzzzzzzzsplit_movie.json", "a+") as fout:
        json.dump(jssssson, fout)
    print('ghi xong json')

    i = 0
    t = datetime.datetime.now()
    label22222222222id = {s:0 for s in label2id.keys()}
    dieeeeeeeeeeemtrichdoan = []
    ########################################################################
    with open('zzprrrre.txt', 'a+', encoding='utf-8') as prrrre:#with open('zzprrrre.txt', 'w', encoding='utf-8') as prrrre:
        for start,stop in silence:
            img = mussssssssssic(r'videooooooooooooo', start, stop, label2id=label2id)
            key, iiiiiiiiiiiiii, diemtrichdoan = kkkkkkkkkkkkkkkkmmmeannnnnnn(X, Y, image_descriptors, BoW, img = img, i = i)#(X, Y, image_descriptors, BoW, img = cv2.imread(img), i = i, start=start, stop=stop)
            label22222222222id[key] += 1
            i += 1
            keyyy = '\n'+str(img)+' mang sắc thái âm thanh của thể loại '+str(key)+'\n'
            print(keyyy)
            prrrre.write(iiiiiiiiiiiiii)
            prrrre.write(keyyy)
            dieeeeeeeeeeemtrichdoan.append((diemtrichdoan, (start, stop)))            
    
    print(label22222222222id)
    dieeeeeeeeeeemtrichdoan.sort(reverse=True, key=myFunc)
    print(dieeeeeeeeeeemtrichdoan)
    global samxxxxxxxinhdep

    for k in range(int(soluong)):
            sss=dieeeeeeeeeeemtrichdoan[k][1][0]
            ssss=dieeeeeeeeeeemtrichdoan[k][1][1]
            print(sss, ssss)
            samxxxxxxxinhdep.append((sss, 'zzzssvcoutput'+str(sss)+'-'+str(ssss)+'.mp4'))
            cutting_theeeeeeeeeee_video(inputtttt = str(filllllllm),
                                        outputttttt = 'zzzssvcoutput'+str(sss)+'-'+str(ssss)+'.mp4',
                                        start = str(sss),
                                        stop = str(ssss-sss),
                                        fps = fps)
##    samxxxxxxxinhdep.sort(key=myFunc)
##    print(samxxxxxxxinhdep)
##
##    with open('zzzssvclisssssssst.txt', 'a+', encoding='utf-8') as fiiiiiiiiiiiiiiina:#with open('zzzlisssssssst.txt', 'w', encoding='utf-8') as fiiiiiiiiiiiiiiina:
##        for samxxxxxxxinhdepppp in samxxxxxxxinhdep:
##            fiiiiiiiiiiiiiiina.write("file '"+samxxxxxxxinhdepppp[1]+"'")
##            fiiiiiiiiiiiiiiina.write("\n")
    print('chương trình đã chạy trong ', datetime.datetime.now()-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzs)

##    val = max(label22222222222id.values())
##    #print(int(val))
##    for labbbbb in label22222222222id.items():
##    ##        print(labbbbb[0]) if int(labbbbb[1])==int(val) else print('haahahehehoho')
##            if int(labbbbb[1])==int(val):
##                theloai = labbbbb[0]
##    print('bộ phim được dự đoán thuộc thể loại ', theloai)
    
##    theloai='samxxxxxxxxxinhdep'
##    while not theloai in label22222222222id.keys():
##        theloai = input('nhập tên thể loại mong muốn trích xuất ')##sssssssssssssssssssss = input("gõ enter")

##    x , sr = rrrreaddata("zzzmergedfile.mp4")
##    display_mel_spectogram(x, sr)
##
##    plt.savefig(os.path.join(os.getcwd(), 'trainingset_audio', theloai, str(datetime.datetime.now()).replace(' ', '.').replace(':', '.')+'.png'))
##    global button4
##    button4.destroy()

def kerrrrrrrras(filllllllm, silence, giiiiiiiiiiioihanthoigian, gioihantren):
    from keras.models import Sequential, load_model
    from scipy.io import wavfile as wav
    import numpy as np

    def extract_feature(file_name, start=None, stop=None):
      
##        try:
            if start and stop:
                audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast', offset=start, duration=stop-start)
            else:
                audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T,axis=0)
            
##        except Exception as e:
##            print("Error encountered while parsing file: ", file)
##            return None, None

            return np.array([mfccsscaled])

    def extract_features(file_name):
       
    ##    try:
            print(file_name)
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T,axis=0)
            return mfccsscaled        
    ##    except Exception as e:
    ##        print("Error encountered while parsing file: ", file)
    ##        return None 
         
    ##    return mfccsscaled

    def print_prediction(file_name, start=None, stop=None):
        prediction_feature = extract_feature(file_name, start, stop) #phát hiện đặc trưng của đối tượng

        predicted_vector = model.predict_classes(prediction_feature) #dự đoán đặc trưng vừa phát hiện thuộc vào lớp nào trong mô hình đã huấn luyện
        predicted_class = le.inverse_transform(predicted_vector) #chuyển đổi từ số đã mã hóa của lớp thành tên lớp
        print("The predicted class is:", predicted_class[0], '\n') 

        predicted_proba_vector = model.predict_proba(prediction_feature) #điểm số của từng lớp trong mô hình đã huấn luyện
        predicted_proba = predicted_proba_vector[0] #điểm số của từng lớp trong mô hình đã huấn luyện
        for i in range(len(predicted_proba)): 
            category = le.inverse_transform(np.array([i])) #chuyển đổi từ số đã mã hóa của lớp thành tên lớp
            print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )
##1512            
        return max(predicted_proba)

    def plot_model_history(model_history, acc='accuracy', val_acc='val_accuracy'):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        import tensorflow as zzss
        if str(zzss.__version__)[0]=='1':
            acc='acc'
            val_acc='val_acc'
        axs[0].plot(range(1, len(model_history.history[acc]) + 1), model_history.history[acc])
        axs[0].plot(range(1, len(model_history.history[val_acc]) + 1), model_history.history[val_acc])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1, len(model_history.history[acc]) + 1), len(model_history.history[acc]) / 10)
        axs[0].legend(['train', 'val'], loc='best')
        axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
        axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
        axs[1].legend(['train', 'val'], loc='best')
##        plt.show()
        plt.savefig('zzzroc.png')

    def create_model(x_train = None, x_test = None, y_train = None, y_test = None, yy = None, le = None, num_epochs = 100):
        from keras.layers import Dense, Dropout, Activation, Flatten
        from keras.layers import Convolution2D, MaxPooling2D
        from keras.optimizers import Adam
        from keras.utils import np_utils
        from sklearn import metrics 

        num_labels = yy.shape[1]
        filter_size = 2

        # Construct model 
        model = Sequential()

        model.add(Dense(256, input_shape=(40,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(num_labels))
        model.add(Activation('softmax'))
    ##
    ##    ##################################################         CNN
    ##    import numpy as np
    ##    from keras.models import Sequential, load_model
    ##    from keras.layers import Dense, Dropout, Activation, Flatten
    ##    from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
    ##    from keras.optimizers import Adam
    ##    from keras.utils import np_utils
    ##    from sklearn import metrics 
    ##
    ##    num_rows = 40
    ##    num_columns = 174
    ##    num_channels = 1
    ##
    ##    x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
    ##    x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)
    ##
    ##    num_labels = yy.shape[1]
    ##    filter_size = 2
    ##
    ##    # Construct model 
    ##    model = Sequential()
    ##    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
    ##    model.add(MaxPooling2D(pool_size=2))
    ##    model.add(Dropout(0.2))
    ##
    ##    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    ##    model.add(MaxPooling2D(pool_size=2))
    ##    model.add(Dropout(0.2))
    ##
    ##    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    ##    model.add(MaxPooling2D(pool_size=2))
    ##    model.add(Dropout(0.2))
    ##
    ##    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    ##    model.add(MaxPooling2D(pool_size=2))
    ##    model.add(Dropout(0.2))
    ##    model.add(GlobalAveragePooling2D())
    ##
    ##    model.add(Dense(num_labels, activation='softmax')) 

    # ### Compiling the model 
    # For compiling our model, we will use the following three parameters: 
    # * Loss function - we will use `categorical_crossentropy`. This is the most common choice for classification. A lower score indicates that the model is performing better.
    # * Metrics - we will use the `accuracy` metric which will allow us to view the accuracy score on the validation data when we train the model. 
    # * Optimizer - here we will use `adam` which is a generally good optimizer for many use cases.

        # Compile the model
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 
        print('model.compile')

            # Display model architecture summary 
        model.summary()

        # Calculate pre-training accuracy 
        score = model.evaluate(x_test, y_test, verbose=0)
        accuracy = 100*score[1]

        print("Pre-training accuracy: %.4f%%" % accuracy)

        # ### Training 
        # 
        # Here we will train the model. 
        # 
        # We will start with 100 epochs which is the number of times the model will cycle through the data. The model will improve on each cycle until it reaches a certain point. 
        # We will also start with a low batch size, as having a large batch size can reduce the generalisation ability of the model. 

        from keras.callbacks import ModelCheckpoint 
        from datetime import datetime 

        num_batch_size = 32

        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_mlp.hdf5', 
                                       verbose=1, save_best_only=True)
        start = datetime.now()

        sssssssssssss=model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)
        plot_model_history(sssssssssssss)

        duration = datetime.now() - start
        print("Training completed in time: ", duration)
        model.save('zzzmy_model')
        
        return model

    import pandas as pd
    import pickle

    fulldatasetpath = os.path.join(os.getcwd(), 'trainingset_audio')
    num_epochs = 80#100#20#

    if not os.path.isfile('zzepoch'+str(num_epochs)+'.pkl'):
        features = []

        for label in os.listdir(fulldatasetpath):
            for sounddd in os.listdir(os.path.join(fulldatasetpath, label)):
                if str(sounddd)[-4:] == '.wav' or str(sounddd)[-4:] == '.mp4':
                    data = extract_features(os.path.join(fulldatasetpath, label, sounddd))
                    features.append([data, label])

        # Convert into a Panda dataframe 
        featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
     
    # Saving the objects:
        with open('zzepoch'+str(num_epochs)+'.pkl', 'wb') as f:
            pickle.dump(featuresdf, f)#[obj0, obj1, obj2], f)
    else: 
    # Getting back the objects:
        with open('zzepoch'+str(num_epochs)+'.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            featuresdf = pickle.load(f)#obj0, obj1, obj2 = pickle.load(f)

    print('Finished feature extraction from ', len(featuresdf), ' files') 

    # ### Convert the data and labels
    from sklearn.preprocessing import LabelEncoder
    from keras.utils import to_categorical

        # Convert features and corresponding classification labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())

        # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y)) 

        # ### Split the dataset

    from sklearn.model_selection import train_test_split 
    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
    print('train_test_split done')
    # ### Test the model 
    # 
    # Here we will review the accuracy of the model on both the training and test data sets.
    if not os.path.exists('zzzmy_model'):
    ##    model = create_model(num_epochs = num_epochs)
        model = create_model(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test, yy = yy, le = le, num_epochs = num_epochs)
    else:
        model = load_model('zzzmy_model')
        model.summary()

    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: ", score[1])

    ##############################################################################
    ##from gioihanthoigian import gioihanthoigiannnnnnnn
    ##silence = gioihanthoigiannnnnnnn(silence=silence, gioihanduoi=giiiiiiiiiiioihanthoigian, gioihantren = gioihantren)
    ##print('giới hạn mỗi đoạn dài ', giiiiiiiiiiioihanthoigian, 'giây, tối đa ', gioihantren, 'giây, chia lại thành ', len(silence), ' đoạn')
    ##############################################################################            
    ##############################################################################
    ### ### Predictions
    ##print_prediction(filllllllm)
##1512                
    diiiiiiiiemtrichdoan=[]
    for start, stop in silence:        
            if stop-start>=giiiiiiiiiiioihanthoigian:#áp dụng giới hạn dưới
                print(start, stop)
##                print_prediction(filllllllm, start, stop)
##1512                
                diemtrichdoan = print_prediction(filllllllm, start, stop)
                diiiiiiiiemtrichdoan.append((diemtrichdoan, (start, stop)))
            
    diiiiiiiiemtrichdoan.sort(reverse=True, key=myFunc)
##    sắp xếp các trích đoạn giảm dần theo số điểm của nó
    global samxxxxxxxinhdep
##    print(diiiiiiiiemtrichdoan)

    for k in range(int(soluong)):
##            print(k, '/', soluong)
            sss=diiiiiiiiemtrichdoan[k][1][0]
            ssss=diiiiiiiiemtrichdoan[k][1][1]
            if ssss-sss>gioihantren:
                ssss=sss+gioihantren
            print(sss, ssss)
            samxxxxxxxinhdep.append((sss, 'zzzkerasoutput'+str(sss)+'-'+str(ssss)+'.mp4'))
            cutting_theeeeeeeeeee_video(inputtttt = str(filllllllm),
                                        outputttttt = 'zzzkerasoutput'+str(sss)+'-'+str(ssss)+'.mp4',
                                        start = str(sss),
                                        stop = str(ssss-sss),
                                        fps = fps)
##    samxxxxxxxinhdep.sort(key=myFunc)
##    print(samxxxxxxxinhdep)
##
##    with open('zzzkeraslisssssssst.txt', 'a+', encoding='utf-8') as fiiiiiiiiiiiiiiina:#with open('zzzlisssssssst.txt', 'w', encoding='utf-8') as fiiiiiiiiiiiiiiina:
##        for samxxxxxxxinhdepppp in samxxxxxxxinhdep:
##            fiiiiiiiiiiiiiiina.write("file '"+samxxxxxxxinhdepppp[1]+"'")
##            fiiiiiiiiiiiiiiina.write("\n")
    print('chương trình đã chạy trong ', datetime.datetime.now()-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzs)
##    global button5
##    button5.destroy()
    
##def ruuunssvc():
##    ssvc(min_silence_len, giiiiiiiiiiioihanthoigian*3, gioihantren*5, silence_thresh, filllllllm, len___silence, label2id, silence, fps, iiiiiiiiiiiiiiiiiiiiiiiiiiiiii)
####    svcsvc(min_silence_len, giiiiiiiiiiioihanthoigian, gioihantren, silence_thresh, filllllllm, len___silence, label2id, silence, fps, iiiiiiiiiiiiiiiiiiiiiiiiiiiiii)
####    kerrrrrrrras(filllllllm, silence, giiiiiiiiiiioihanthoigian, gioihantren)
##    
##def ruuunkerrrrrrrras():
####    ssvc(min_silence_len, giiiiiiiiiiioihanthoigian, gioihantren, silence_thresh, filllllllm, len___silence, label2id, silence, fps, iiiiiiiiiiiiiiiiiiiiiiiiiiiiii)
####    svcsvc(min_silence_len, giiiiiiiiiiioihanthoigian, gioihantren, silence_thresh, filllllllm, len___silence, label2id, silence, fps, iiiiiiiiiiiiiiiiiiiiiiiiiiiiii)
##    kerrrrrrrras(filllllllm, silence, giiiiiiiiiiioihanthoigian, gioihantren)
##    
##def ruuunsvcsvc():
####    ssvc(min_silence_len, giiiiiiiiiiioihanthoigian, gioihantren, silence_thresh, filllllllm, len___silence, label2id, silence, fps, iiiiiiiiiiiiiiiiiiiiiiiiiiiiii)
##    svcsvc(min_silence_len, giiiiiiiiiiioihanthoigian, gioihantren, silence_thresh, filllllllm, len___silence, label2id, silence, fps, iiiiiiiiiiiiiiiiiiiiiiiiiiiiii)
####    kerrrrrrrras(filllllllm, silence, giiiiiiiiiiioihanthoigian, gioihantren)

def viiideo():
    global samxxxxxxxinhdep
    samxxxxxxxinhdep.sort(key=myFunc)
    print(samxxxxxxxinhdep)

    with open('zzzlisssssssst.txt', 'w', encoding='utf-8') as fiiiiiiiiiiiiiiina:#with open('zzzlisssssssst.txt', 'w', encoding='utf-8') as fiiiiiiiiiiiiiiina:
            for samxxxxxxxinhdepppp in samxxxxxxxinhdep:
                fiiiiiiiiiiiiiiina.write("file '"+samxxxxxxxinhdepppp[1]+"'")
                fiiiiiiiiiiiiiiina.write("\n")

    os.system('ffmpeg -f concat -safe 0 -i zzzlisssssssst.txt -c:a copy -c:v copy zzzmergedfile.mp4')# -filter:v fps=fps='+str(fps)+ ' zzzmergedfile.mp4')
##    os.system('ffmpeg -hwaccel auto -hide_banner -vsync 1 -f concat -safe 0 -i zzzlisssssssst.txt -c:v h264_amf -c:a copy -gpu all zzzmergedfile.mp4')
    print('done')

def remooooooooooove_audio(inputtttt = 'file.mp4',
                           outputttttt = 'ffffffffffffile.mp4'):
    os.system('ffmpeg -i '+ inputtttt+ ' -c copy -an '+ outputttttt)#remove audio
##    os.system('ffmpeg -hwaccel auto -hide_banner -vsync 1 -i '+ inputtttt+ ' -c:v h264_amf -gpu all -an '+ outputttttt)#
##    https://www.google.com.vn/search?sxsrf=ALeKk00muLzc3oSqf7fT_UMAbb9QZLCdqg%3A1605059954928&ei=ckWrX_GiOLySr7wPpomaMA&q=moviepy+%22without_audio%22&oq=moviepy+%22without_audio%22&gs_lcp=CgZwc3ktYWIQAzoECCMQJzoHCCMQsAIQJzoGCAAQFhAeUO1JWLiMAWDvkQFoAHAAeACAAYABiAG_A5IBAzEuM5gBAKABAaoBB2d3cy13aXrAAQE&sclient=psy-ab&ved=0ahUKEwjxo9OjsvnsAhU8yYsBHaaEBgYQ4dUDCA0&uact=5

def changggggggggee_resolution(inputtttt = 'file.mp4',
                                outputttttt = 'ffffffffffffile.mp4',
                               fps = 24):
    os.system('ffmpeg -i '+ inputtttt+ """ -vf "fps="""+str(fps)+ """" -s 1280x1280 -c:v copy -c:a copy """+ outputttttt)#
##    os.system('ffmpeg -hwaccel auto -hide_banner -vsync 1 -i '+ inputtttt+ """ -vf "fps="""+str(fps)+ """" -s 1280x1280 -c:v h264_amf -c:a copy -gpu all """+ outputttttt)#
##https://stackoverflow.com/questions/11779490/how-to-add-a-new-audio-not-mixing-into-a-video-using-ffmpeg

def changggggggggee_aspectratio(inputtttt = 'file.mp4',
                                outputttttt = 'ffffffffffffile.mp4',
                               fps = 24):
    os.system('ffmpeg -i '+ inputtttt+ """ -vf "fps="""+str(fps)+ """" aspect 1:1 """+ outputttttt)#
##    os.system('ffmpeg -hwaccel auto -hide_banner -vsync 1 -i '+ inputtttt+ """ -c:v h264_amf -c:a copy -vf "fps="""+str(fps)+ """" aspect 1:1 -gpu all """+ outputttttt)#
##https://stackoverflow.com/questions/11779490/how-to-add-a-new-audio-not-mixing-into-a-video-using-ffmpeg

def convvvvvvvvvert_video(inputtttt = 'file.mov',
                         outputttttt = 'output.mp4',
                         deleteeeee = None):
    os.system('ffmpeg -i '+ inputtttt+ ' '+ outputttttt)
##    os.system('ffmpeg -hwaccel auto -hide_banner -vsync 1  -i '+ inputtttt+ ' -c:v h264_amf -c:a copy -gpu all '+ outputttttt)
    if deleteeeee:#convert and replace video
        os.remove(os.path.join(os.getcwd(), inputtttt))

def cutting_theeeeeeeeeee_video(inputtttt = 'file.mp4',
                         outputttttt = 'output.mp4',
                                start = '00:00:03',
                                stop = '00:00:08',#length
                                fps = 24):
    os.system('ffmpeg -hide_banner -vsync 1 -i '+ inputtttt+ ' -ss '+ start+ ' -t '+ stop+ ' -c:a copy -y '+ 'z'+ outputttttt)
##    os.system('ffmpeg -i '+ inputtttt+ ' -ss '+ start+ ' -t '+ stop+ ' -c:v copy -c:a copy zzzzzz'+ outputttttt)
####    ssssssssssssssssssssssssssssss='ffmpeg -i zzzzzz'+ outputttttt+ """ -preset slow -vf "fps="""+str(fps)+""",fade=type=in:duration=2,fade=type=out:duration=2:start_time="""+str(int(stop)-2)+"""" -c:a copy """+outputttttt#-filter:v fps=fps="""+str(fps)+' '+outputttttt
####    ssssssssssssssssssssssssssssss='ffmpeg -hwaccel auto -hide_banner -vsync 1 -i zzzzzz'+ outputttttt+ """ -preset slow -vf "fps="""+str(fps)+""",fade=type=in:duration=2,fade=type=out:duration=2:start_time="""+str(int(stop)-2)+"""" -c:v h264_amf -c:a copy -gpu all """+outputttttt
    ssssssssssssssssssssssssssssss='ffmpeg -i z'+ outputttttt+ """ -vf "fps="""+str(fps)+""",fade=type=in:duration=2,fade=type=out:duration=2:start_time="""+str(int(stop)-2)+"""" -c:a copy -y """+outputttttt#-acodec copy 
    os.system(ssssssssssssssssssssssssssssss)
    os.remove('z'+ outputttttt)
##    return ssssssssssssssssssssssssssssss

def extract_tttttttthe_audio(inputtttt = os.path.join(os.getcwd(), 'videooooooooooooo', 'videooooooooooooo', 'file.mp4'),
                             outputttttt = os.path.join(os.getcwd(), 'videooooooooooooo', 'videooooooooooooo', 'file.mp3'),
                             ssstart = None,
                             ssstop = None):
    ssssssssssssssssssssssssss='ffmpeg -i '+ inputtttt + ' -ss '+ str(ssstart)+ ' -to '+ str(ssstop)+ ' -y '+outputttttt
##    ssssssssssssssssssssssssss='ffmpeg -hwaccel auto -hide_banner -vsync 1 -i '+ inputtttt + ' -ss '+ str(ssstart)+ ' -to '+ str(ssstop)+ ' -c:a copy -gpu all '+outputttttt
    os.system(ssssssssssssssssssssssssss)

zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzs, samxxxxxxxinhdep = datetime.datetime.now(), []
button1 = tk.Button(
    root,
    text="Xóa bỏ\n âm thanh",
    width=25,
    height=5,
    bg="green",
    fg="white",
    command=lambda: remooooooooooove_audio(inputtttt = filllllllm,
                           outputttttt = 'file.mp4')
)
##button1.place(x=75, y=69)#button.pack()
##button1.pack(fill=tk.BOTH, side=tk.RIGHT, expand=True)#tk.X, tk.Y, tk.BOTH
button1.grid(row=0,column=5,sticky='EWNS')

button2 = tk.Button(
    root,
    text="Thay đổi\n độ phân giải",
    width=25,
    height=5,
    bg="green",
    fg="white",
    command=lambda: changggggggggee_resolution(inputtttt = filllllllm,
                           outputttttt = 'file.mp4')
)
##button2.place(x=75, y=69)#button.pack()
##button2.pack(fill=tk.BOTH, side=tk.RIGHT, expand=True)#tk.X, tk.Y, tk.BOTH
button2.grid(row=1,column=5,sticky='EWNS')

button3 = tk.Button(
    root,
    text="Thay đổi\ntỷ lệ\nkhung hình",
    width=25,
    height=5,
    bg="green",
    fg="white",
    command=lambda: changggggggggee_aspectratio(inputtttt = filllllllm,
                           outputttttt = 'file.mp4')
)
##button3.place(x=75, y=69)#button.pack()
##button3.pack(fill=tk.BOTH, side=tk.RIGHT, expand=True)#tk.X, tk.Y, tk.BOTH
button3.grid(row=2,column=5,sticky='EWNS')

cam = cv2.VideoCapture(filllllllm)
fps = cam.get(cv2.CAP_PROP_FPS)
frame_count = cam.get(cv2.CAP_PROP_FRAME_COUNT)
thhhhhhhhhhhhhhhhhhoigiiiiiiiiiiiiiiiiiiian = [(0, (frame_count/fps)/3), ((frame_count/fps)/3, (frame_count/fps)/3*2), ((frame_count/fps)/3*2, (frame_count/fps))]

##########################################################xử lí dữ liệu train
import numpy as np
t = datetime.datetime.now()
label2id = labellllll2id()

##lưu đặc trưng âm thanh dưới dạng ảnh
mussssssssssic()

if not os.path.isfile('zzbow_dictionaryyyyyyyyy.pkl'):
    X, Y = read_data(label2id)
    image_descriptors = extract_sift_features(X)
    
    print('số các list chứa các SIFT keypoints descriptor ứng với từng ảnh: ', len(image_descriptors))
    print('#######################'+str(image_descriptors[0][1].shape))
    for i in range(len(image_descriptors)):
        print('Image {} has {} descriptors'.format(i, len(image_descriptors[i])))

    all_descriptors = []
    for descriptors in image_descriptors:
        if descriptors is not None:
            for des in descriptors:
                all_descriptors.append(des)
    print('Total number of descriptors: %d' %(len(all_descriptors)))

# Lưu từ điển phục vụ cho việc sử dụng sau này:
    num_clusters = 50#15
    
    BoW = kmeans_bow(all_descriptors, num_clusters)
    pickle.dump([X, Y, image_descriptors, BoW], open('zzbow_dictionaryyyyyyyyy.pkl', 'wb'))
else:
##    BoW = pickle.load(open('zzbow_dictionaryyyyyyyyy.pkl', 'rb'))
    X, Y, image_descriptors, BoW = pickle.load(open('zzbow_dictionaryyyyyyyyy.pkl', 'rb'))
print('thời gian máy xử lí dữ liệu train: ', datetime.datetime.now()-t)

##################################################xử lí video cần trích xuất
def llllllllllloop(iiiiiiiiiiiiiiiiiiiiiiiiiiiiii, ssstart, ssstop):
    global min_silence_len, giiiiiiiiiiioihanthoigian, gioihantren, silence_thresh, filllllllm, len___silence, soluong
    t=datetime.datetime.now()
    print('########################################################', iiiiiiiiiiiiiiiiiiiiiiiiiiiiii, ' ', t)
    print(ssstart, ssstop)
    mmmmppppppppppba = os.path.join(os.getcwd(), 'videooooooooooooo', 'videooooooooooooo', 'zzzzzz'+ str(iiiiiiiiiiiiiiiiiiiiiiiiiiiiii)+'.mp3')
##    os.system('ffmpeg -i '+ filllllllm+ ' -ss '+ str(zzzzzzzzzzzzzs[0])+ ' -t '+ str(zzzzzzzzzzzzzs[1])+ ' '+ ouuuuuuuuuuuuutput)#
    if not os.path.isfile('zz_MinSilenceLen_'+str(min_silence_len)+'_thresh_-'+str(abs(silence_thresh))+'zzzzzz'+ str(iiiiiiiiiiiiiiiiiiiiiiiiiiiiii)+'.csv'):
        silllence, i, tt = [], 0, datetime.datetime.now()        
        extract_tttttttthe_audio(inputtttt = filllllllm,#ouuuuuuuuuuuuutput,
                                 outputttttt = mmmmppppppppppba,
                                 ssstart=round(ssstart),
                                 ssstop=round(ssstop))    
##        os.system('ffmpeg -i '+ filllllllm + ' -ss '+ str(ssstart)+ ' -t '+ str(ssstop)+ ' '+str(mmmmppppppppppba))
        silence, i, min_silence_len, silence_thresh=detectttttttttttsilence(filllllllm = mmmmppppppppppba,
                                           min_silence_len = min_silence_len,
                                           silence_thresh = silence_thresh,
                                           len___silence=len___silence)#silllence
        
        print('detect silence in ', str(datetime.datetime.now()-tt), 'second and ', i, ' times')
        os.remove(mmmmppppppppppba)
##        print(silence)
        with open('zz_MinSilenceLen_'+str(min_silence_len)+'_thresh_-'+str(abs(silence_thresh))+'zzzzzz'+ str(iiiiiiiiiiiiiiiiiiiiiiiiiiiiii)+'.csv', 'a+') as s:
            s.write('start_time,length,rename_to\n')
            
            for zzzzzzzzzzzzzzzzszzzzz, (starrrt, stttop) in enumerate(silence):
                starrrt+=round(ssstart)
                stttop+=round(ssstart)
                s.write(str(starrrt)+','+str(stttop)+',vid'+str(starrrt)+'-'+str(stttop)+'\n')
                silence[zzzzzzzzzzzzzzzzszzzzz]=(starrrt, stttop)
            print(silence)
    else:
        with open('zz_MinSilenceLen_'+str(min_silence_len)+'_thresh_-'+str(abs(silence_thresh))+'zzzzzz'+ str(iiiiiiiiiiiiiiiiiiiiiiiiiiiiii)+'.csv') as manifest_file:
            silence = []
            config = csv.DictReader(manifest_file)
            for video_config in config:
                start = int(video_config["start_time"])
                stop = int(video_config["length"])
                silence.append((start,stop))
    
    button4 = tk.Button(
                root,
                text="Bắt đầu\nphân tích\nssvc!",
                width=25,
                height=5,
                bg="blue",
                fg="yellow",
                command= lambda c=silence: ssvc(min_silence_len, giiiiiiiiiiioihanthoigian*3, gioihantren*5, silence_thresh, filllllllm, len___silence, label2id, c, fps, iiiiiiiiiiiiiiiiiiiiiiiiiiiiii)#ruuunssvc
            )
    ##button4.place(x=75, y=69)#button.pack()
    ##button4.pack(fill=tk.BOTH, side=tk.RIGHT, expand=True)#tk.X, tk.Y, tk.BOTH
    button4.grid(row=iiiiiiiiiiiiiiiiiiiiiiiiiiiiii,column=1)

##    button5 = tk.Button(
##                root,
##                text="Bắt đầu\nphân tích\nkerrrrrrrras!",
##                width=25,
##                height=5,
##                bg="blue",
##                fg="yellow",
##                command= lambda c=silence: kerrrrrrrras(filllllllm, c, giiiiiiiiiiioihanthoigian, gioihantren)#ruuunkerrrrrrrras
##            )
##    ##button5.place(x=75, y=69)#button.pack()
##    ##button5.pack(fill=tk.BOTH, side=tk.RIGHT, expand=True)#tk.X, tk.Y, tk.BOTH
##    button5.grid(row=iiiiiiiiiiiiiiiiiiiiiiiiiiiiii,column=2)#,sticky='EWNS')

    button6 = tk.Button(
                root,
                text="Bắt đầu\nphân tích\nsvcsvc!",
                width=25,
                height=5,
                bg="blue",
                fg="yellow",
                command= lambda c=silence: svcsvc(min_silence_len, giiiiiiiiiiioihanthoigian, gioihantren, silence_thresh, filllllllm, len___silence, label2id, c, fps, iiiiiiiiiiiiiiiiiiiiiiiiiiiiii)
            )
    ##button6.place(x=75, y=69)#button.pack()
    ##button6.pack(fill=tk.BOTH, side=tk.RIGHT, expand=True)#tk.X, tk.Y, tk.BOTH
    button6.grid(row=iiiiiiiiiiiiiiiiiiiiiiiiiiiiii,column=3)#,sticky='EWNS')
    
##    ssvc(min_silence_len, giiiiiiiiiiioihanthoigian, gioihantren, silence_thresh, filllllllm, len___silence, label2id, silence, fps, iiiiiiiiiiiiiiiiiiiiiiiiiiiiii)
##    svcsvc(min_silence_len, giiiiiiiiiiioihanthoigian, gioihantren, silence_thresh, filllllllm, len___silence, label2id, silence, fps, iiiiiiiiiiiiiiiiiiiiiiiiiiiiii)
    kerrrrrrrras(filllllllm, silence, giiiiiiiiiiioihanthoigian, gioihantren)
    print(iiiiiiiiiiiiiiiiiiiiiiiiiiiiii, ' ', datetime.datetime.now()-t)
    
for iiiiiiiiiiiiiiiiiiiiiiiiiiiiii, (ssstart, ssstop) in enumerate(thhhhhhhhhhhhhhhhhhoigiiiiiiiiiiiiiiiiiiian):
    llllllllllloop(iiiiiiiiiiiiiiiiiiiiiiiiiiiiii, ssstart, ssstop)#Pool(processes=multiprocessing.cpu_count())
##pool = Pool(processes=3)#(os.cpu_count())                         # Create a multiprocessing Pool
##pool.starmap(llllllllllloop, enumerate(thhhhhhhhhhhhhhhhhhoigiiiiiiiiiiiiiiiiiiian))  # process data_inputs iterable with pool
            
button7 = tk.Button(
                root,
                text="Trích xuất\nvideo!",
                width=25,
                height=5,
                bg="blue",
                fg="yellow",
                command= viiideo
            )
##button7.place(x=75, y=69)#button.pack()
##button7.pack(fill=tk.BOTH, side=tk.RIGHT, expand=True)#tk.X, tk.Y, tk.BOTH
button7.grid(row=0,column=4)#,sticky='EWNS')

            
button8 = tk.Button(
                root,
                text="KẾT THÚC CHƯƠNG TRÌNH!",
                width=25,
                height=5,
                bg="blue",
                fg="yellow",
                command= root.destroy
            )
##button8.place(x=75, y=69)#button.pack()
##button8.pack(fill=tk.BOTH, side=tk.RIGHT, expand=True)#tk.X, tk.Y, tk.BOTH
button8.grid(row=2,column=4)#,sticky='EWNS')

root.mainloop()#run the Tkinter event loop


##pyinstaller --add-data D:/Python/Lib/site-packages/librosa/util/example_data;librosa/util/example_data --onefile tkkkkkkkkkkkkkinter.py
