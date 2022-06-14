import pandas as pd
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

from sklearn import tree
from matplotlib import pyplot as plt

data = load_breast_cancer()
dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])
dataset

temp_y = data['target']
x = dataset.copy()

y = []

initial = {"cancerous": 0, "not cancerous": 0}
result = {"cancerous": 0, "not cancerous": 0}

for i in range(len(temp_y)):
    y.append(temp_y[i])
    
for i in range(len(y)):
    if y[i] == 0:
        y[i] = "Not Cancerous"
        initial["not cancerous"] += 1
    else:
        y[i] = "Cancerous"
        initial["cancerous"] += 1

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.45)

clf = DecisionTreeClassifier(ccp_alpha=0.01)
clf = clf.fit(X_train, y_train)

clf.get_params()

predictions = clf.predict(X_test)
print(predictions)

for i in range(len(predictions)):
    if predictions[i] == 'Cancerous':
        result["cancerous"] += 1
    else:
        result["not cancerous"] += 1

y_test

accuracy_score(y_test, predictions)

confusion_matrix(y_test, predictions, labels=['Cancerous', 'Not Cancerous'])

print('Cancerous Precision: ' + str(precision_score(y_test, predictions, pos_label='Cancerous')))
print('Not Cancerous Precision: ' + str(precision_score(y_test, predictions, pos_label='Not Cancerous')))

print('Cancerous Recall Score: ' + str(recall_score(y_test, predictions, pos_label='Cancerous')))
print('Not Cancerous Recall Score: ' + str(recall_score(y_test, predictions, pos_label='Not Cancerous')))

print(classification_report(y_test, predictions, target_names=['Cancerous', 'Not Cancerous']))

print("initial:")
print(initial)

print("result:")
print(result)

print(len(y_test))

clf.feature_importances_

feature_importances = pd.DataFrame(clf.feature_importances_, index=x.columns).sort_values(0, axis=0, ascending=False)
feature_importances

feature_importances.head(10).plot(kind='bar')

fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(clf,
                    feature_names=x.columns,
                    class_names={0: "Cancerous", 1: "Not Cancerous"},
                    filled=True,
                    fontsize=14)

