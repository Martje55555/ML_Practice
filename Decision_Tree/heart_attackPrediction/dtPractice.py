import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv('heart.csv')
data.info()
dataset = pd.DataFrame(data=data, columns=data.columns)
dataset

y = data['output']
x = dataset.copy()
x = x.drop(columns=['output'])
print(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

clf.get_params()

predictions = clf.predict(X_test)
print(predictions)

y_test

accuracy_score(y_test, predictions)

print(len(y_test))