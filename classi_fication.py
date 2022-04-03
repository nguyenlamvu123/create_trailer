##KNN
##Decision Tree
##Random Forest
##Linear SVM 
##Non-linear SVM 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

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
        param = {'C': [0.5, 1, 5], 'kernel': ['poly', 'rbf', 'sigmoid']}
        return self.fiiit(knn, param)
    def fiiit(self, knn, param):
        print(knn)
        knnn = knn
        if bestParams==True:
            from sklearn.model_selection import GridSearchCV
            knnn = GridSearchCV(
                estimator=knn,
                param_grid=param,
                cv=3, n_jobs=4
                )
            print('knnn.best_params_: ', knnn.best_params_)
        knnn.fit(self.x_train, self.y_train)
        print(knnn.score(self.x_train, self.y_train))
        print(knnn.score(self.x_test, self.y_test))
s=Clsification()
