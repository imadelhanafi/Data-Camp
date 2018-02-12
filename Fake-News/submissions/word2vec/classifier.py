from sklearn.linear_model import LogisticRegressionCV
import numpy as np
from sklearn.base import BaseEstimator
from xgboost.sklearn import XGBClassifier

 
def score(p):
    if p[1] > 0.5:
        return [0, 0, 0, 0, 1, 0]
    else:
        if p[0] > 1:
            return [1, 0, 0, 0, 0, 0]
        else:
            if p[0] < 0.5: return [0, 0, 1, 0, 0, 0]
            else: return [0, 1, 0, 0, 0, 0]
 
 
class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = LogisticRegressionCV(Cs = [0.16], cv = 5) 
        #self.clf = XGBClassifier()

    def fit(self, X, y):
        self.clf.fit(X, (y>2).astype(int))
 
    def predict(self, X):
        y_pred = self.clf.predict_proba(X)
        return np.array([np.argmax(score(p)) for p in y_pred])
 
    def predict_proba(self, X):
        y_pred = self.clf.predict_proba(X)
        return np.array([score(p) for p in y_pred])
