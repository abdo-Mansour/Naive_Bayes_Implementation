import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NaiveBayes:
    def __init__(self):
        '''
        X: pandas DataFrame
        y: pandas Series
        '''
        self.X_df = None
        self.y_df = None
        self.X = None
        self.y = None
        self.n = 0

        self.features = None
        self.classes = None
        self.class_prob = {}
        self.feature_prob = {}

    def print_class_prob_table(self):
        '''
        print the class probability table
        '''
        for c in self.classes:
            print(f'P({c}) = {self.class_prob[c]}')
    
    def print_feature_prob_table(self):
        '''
        print the feature probability table
        '''

        for f in self.features:
            print(f'Feature: {f}')
            print(self.feature_prob[f])

    def cal_class_prob(self):
        '''
        calculate the class probability
        '''        
        for c in self.classes:
            self.class_prob[c] = np.sum(self.y == c) / len(self.y)    
    
    def cal_likelihood(self):
        '''
        calculate the likelihood of each feature
        '''
        for f in self.features:
            feature_values = np.unique(self.X_df[f].values)
            feature_df = pd.DataFrame(index=feature_values, columns=self.classes)
            for fv in feature_values:
                for c in self.classes:
                    feature_df.at[fv , c] = np.sum((self.X_df[f] == fv) & (self.y_df == c))/np.sum(self.y_df == c)
                
            self.feature_prob[f] = feature_df
        
    
    def fit(self,X,y):
        '''
        X: pandas DataFrame
        y: pandas Series
        '''
        self.X_df = X
        self.y_df = y
        self.X = X.values
        self.y = y.values
        self.n = len(self.y)
        self.features = X.columns
        self.classes = np.unique(y.values)
        self.cal_class_prob()
        self.cal_likelihood()


    def predict(self,X):
        '''
        X: pandas DataFrame
        y: pandas Series

        return: pandas Series
        '''
        y_pred = []
        feature_values = np.unique(X.values)
        X_v = X.values
        for x in X_v:
            prob = {}
            for c in self.classes:
                prob[c] = self.class_prob[c]
                for i in range(len(x)):
                    prob[c] *= self.feature_prob[self.features[i]].at[x[i] , c]
            y_pred.append(max(prob, key=prob.get))
        return pd.Series(y_pred)

data = pd.read_csv('tennis.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
test_df = pd.read_csv('tennis_test.csv')
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]




model = NaiveBayes()
model.fit(X, y)

model.print_class_prob_table()
print("---------------------")
model.print_feature_prob_table()
print("---------------------")


from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print(y_pred)
print('Acc:', accuracy_score(y_test.values, y_pred.values))