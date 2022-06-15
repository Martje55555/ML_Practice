import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

data = pd.read_csv('das.csv')
dataset = pd.DataFrame(data=data, columns=data.columns)
dataset

y = data['Target']
x = dataset.copy()
x = x.drop(columns=['Target'])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)


clf = DecisionTreeClassifier(ccp_alpha=0.01)
clf = clf.fit(X_train, y_train)
clf.get_params()

predictions = clf.predict(X_test)
predictions

accuracy_score(y_test, predictions)
