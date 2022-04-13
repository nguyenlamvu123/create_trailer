##KNN
##Decision Tree
##Random Forest
##Linear SVM 
##Non-linear SVM 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA#Principal Component Analysis

class Prepar_dataset:
    def __init__(
        self,
        path=None,
        ):
        self.path = path
        self.features = []
        self.labels = []

    def giiist(self):
        import gist#https://github.com/tuttieee/lear-gist-python
        cnt = -1

        for c in os.listdir(self.path):
            cnt += 1
            for file_name in os.listdir(
                os.path.join(
                    self.path,
                    c,
                    )
                ):
                file_path = os.path.join(self.path,c,file_name)
                img = cv2.imread(file_path)
                self.labels.append(cnt)
                self.features.append(gist.extract(img))
        
##        X_pan1 = pd.DataFrame(self.features)
##        X_pan1["labels"] = self.labels
##        X_pan1.to_csv("UCMerced_LandUse_PCA.csv")

    def PrincipalComponentAnalysis(self):
        """giảm số chiều dữ liệu"""
        scaled_data = StandardScaler(
            ).fit_transform(self.features)# chuẩn hóa dữ liệu 
        pca = PCA()# giảm só chiều dữ liệu 
        pca.fit(scaled_data)
        print(pca.explained_variance_ratio_)

        pca1 = PCA(n_components=43)
##        data1 = pca1.fit_transform(scaled_data)
        self.features = pca1.fit_transform(scaled_data)

##        X_pan = pd.DataFrame(data1)
##        X_pan["labels"] = self.labels
##        X_pan.to_csv("UCMerced_LandUse.csv")
##        """
##        UCMerced_LandUse.csv: dataset without PCA
##        UCMerced_LandUse_PCA.csv: dataset with PCA
##        """
##        data = pd.read_csv('UCMerced_LandUse.csv')
##        features, labels = data.iloc[:,0:44], data.iloc[:,44]

class Clsification:
    """Hầu hết các class phân loại trong sklearn đều có những phương thức sau:

    fit(X, y): Fit the model using X as training data and y as target values
    predict(X): Predict the class labels for the provided data
    predict_proba(X): Return probability estimates for the test data X (class SVM.SVC không có phương thức này)
    score(X, y): Returns the mean accuracy on the given test data and labels
    """
    def __init__(
        self,
        features=None,
        labels=None,
        bestParams=False,
        ):
        if features is None and labels is None:
            print(self.__doc__)
        else:
            self.features=features
            self.labels=labels
            from sklearn.model_selection import train_test_split
            self.x_train, self.x_test, self.y_train, self.y_test\
                          = train_test_split(
                              self.features,
                              self.labels,
                              test_size=0.2,
                              )
            print('Number of training data: ',
                  self.x_train.shape[0])
            print('Number of testing data: ',
                  self.x_test.shape[0])
    def KNearestNeightbors(self):
        knn = KNeighborsClassifier(
            n_neighbors=1,
            algorithm='kd_tree'
            )
        param = {'n_neighbors': [1, 2]}
        return self.fiiit(knn, param)
    def DecisionTree(self):
        knn = DecisionTreeClassifier()
        param = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'min_samples_leaf': [1, 5, 10]}
        return self.fiiit(knn, param)
    def SupportVectorMachines(self):
        knn = SVC(C = 20)
        param = {'C': [0.5, 1, 5, 20, 100, 500], 'kernel': ['poly', 'rbf', 'sigmoid']}
        return self.fiiit(knn, param)
    def fiiit(self, knn, param):
        print(knn)        
        if bestParams==True:
            from sklearn.model_selection import GridSearchCV
            knnn = GridSearchCV(
                estimator=knn,
                param_grid=param,
                cv=3, n_jobs=4
                )
            print('knnn.best_params_: ', knnn.best_params_)
        else:
            knnn = knn
        knnn.fit(self.x_train, self.y_train)
        print(knnn.score(self.x_train, self.y_train))
        print(knnn.score(self.x_test, self.y_test))
##s=Clsification()
s=Prepar_dataset()
print(s.PrincipalComponentAnalysis.__doc__)
