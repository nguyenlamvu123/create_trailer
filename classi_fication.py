##KNN
##Decision Tree
##Random Forest
##Linear SVM 
##Non-linear SVM 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class Clsification:
    def __init__(
        self,
        features,
        labels,
        ):
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
        return KNeighborsClassifier(
            n_neighbors=1,
            algorithm='kd_tree'
            )
    def DecisionTree(self):
        return DecisionTreeClassifier()
    def SupportVectorMachines(self):
        return SVC(C = 20)
    def fiiit(self, knn)
        print(knn)
        knn.fit(self.x_train, self.y_train)
        print(knn.score(self.x_train, self.y_train))
        print(knn.score(self.x_test, self.y_test))
