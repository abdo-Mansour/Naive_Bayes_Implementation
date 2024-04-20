## Naive Bayes Classifier Implementation

Naive Bayes classifier implementation for categorical data.

### Attributes
- **X_df (pandas DataFrame):** Input features.
- **y_df (pandas Series):** Target labels.
- **X (numpy array):** Numpy array representation of input features.
- **y (numpy array):** Numpy array representation of target labels.
- **n (int):** Number of samples in the dataset.
- **features (pandas Index):** Names of the features.
- **classes (numpy array):** Unique classes in the target labels.
- **class_prob (dict):** Dictionary to store class probabilities.
- **feature_prob (dict):** Dictionary to store feature likelihoods.

### Methods

#### `print_class_prob_table()`:
Print the class probability table.

#### `print_feature_prob_table()`:
Print the feature probability table.

#### `cal_class_prob()`:
Calculate the class probability.

#### `cal_likelihood()`:
Calculate the likelihood of each feature.

#### `fit(X, y)`:
Fit the Naive Bayes classifier to the training data.
- **X (pandas DataFrame):** Input features.
- **y (pandas Series):** Target labels.

#### `predict(X)`:
Predict the class labels for the input data.
- **X (pandas DataFrame):** Input features.
- **Returns:** y_pred (pandas Series) - Predicted class labels.

### Example Usage

```python
data = pd.read_csv('tennis.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
test_df = pd.read_csv('tennis_test.csv')
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# Train the model
model = NaiveBayes()
model.fit(X, y)

# Print class probabilities and feature probabilities
model.print_class_prob_table()
print("---------------------")
model.print_feature_prob_table()
print("---------------------")

# Test the model
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print(y_pred)
print('Acc:', accuracy_score(y_test.values, y_pred.values))
```