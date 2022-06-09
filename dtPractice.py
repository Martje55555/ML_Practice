import pandas as pd
from sklearn.datasets import load_breast_cancer

data = pd.read_csv('heart.csv')
#data.info()
dataset = pd.DataFrame(data=data, columns=data.columns)
dataset

from sklearn.model_selection import train_test_split
y = data['output']
x = dataset.copy()
x = x.drop(columns=['output'])
print(x)

#y = []
# initial = {"cancerous": 0, "not cancerous": 0}
# result = {"cancerous": 0, "not cancerous": 0}
# for i in range(len(temp_y)):
#     y.append(temp_y[i])
    
# for i in range(len(y)):
#     if y[i] == 0:
#         y[i] = "Not Cancerous"
#         initial["not cancerous"] += 1
#     else:
#         y[i] = "Cancerous"
#         initial["cancerous"] += 1


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

clf.get_params()

predictions = clf.predict(X_test)
print(predictions)

# for i in range(len(predictions)):
#     if predictions[i] == 'Cancerous':
#         result["cancerous"] += 1
#     else:
#         result["not cancerous"] += 1

y_test

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

print(len(y_test))