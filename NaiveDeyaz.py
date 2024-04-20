import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NaiveZeyad:
	def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
		'''
		X: pandas DataFrame
		y: pandas Series
		'''		
		self.X, self.y = X, y
		self.df = X;
		self.df[y.name] = y
		self.n = len(X);

		self.classes = self.y.unique()
		self.n_classes = len(self.classes)
		self.features = [c for c in self.X.columns if c != self.y.name]
		self.likelihoods = self._compute_likelihoods()
	def _compute_likelihoods(self):
		classes_count = self.y.value_counts()
		likelihoods = {}
		for labelclass in self.classes:
			likelihoods[(labelclass,)] = classes_count[labelclass] / len(self.df)
			for feature in self.X.columns:
				for value in self.X[feature].unique():
					n = len(self.df[(self.df[feature] == value) & (self.df[self.y.name] == labelclass)])
					likelihoods[(f'{feature}={value}', labelclass)] = n / classes_count[labelclass]
		return likelihoods
	def print_likelihoods(self):
		for key,value in self.likelihoods.items():
			print(f'P{key}= {value}')
	def _calculate_probability(self, samplerow):
		res = {}
		for labelclass in self.classes:
			accumulator = self.likelihoods[(labelclass,)]
			for feature in self.features:
				accumulator *= self.likelihoods[(f'{feature}={samplerow[feature]}', labelclass)]
			res[labelclass] = accumulator;
		return res
	
	def predict(self, X_test):
		y_pred = []
		for _, row in X_test.iterrows():
			pred =self._calculate_probability(row)
			y_pred.append(max(pred, key=pred.get))
		return pd.Series(y_pred)

data = pd.read_csv('tennis.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
test_df = pd.read_csv('tennis_test.csv')
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]
naive = NaiveZeyad(X, y)
naive.print_likelihoods()
print(naive.predict(X_test))