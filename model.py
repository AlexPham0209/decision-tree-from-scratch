import os
import pandas
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

PATH = os.path.join('data', 'credit.csv')
df = pandas.read_csv(PATH)
temp = df.copy()

df['Credit History'] = df['Credit History'].astype('category').cat.codes
df['Debt'] = df['Debt'].astype('category').cat.codes
df['Collateral'] = df['Collateral'].astype('category').cat.codes
df['Credit Risk?'] = df['Credit Risk?'].astype('category').cat.codes
print(df)

temp['Credit History'] = temp['Credit History'].map({'bad': 0, 'good': 1, 'unknown': 2})
temp['Debt'] = temp['Debt'].map({'high': 0, 'low': 1})
temp['Collateral'] = temp['Collateral'].map({'adequate': 0, 'none': 1})
temp['Credit Risk?'] = temp['Credit Risk?'].map({'high': 0, 'low': 1})
print(temp)

features = ['Credit History', 'Debt', 'Collateral']
X = df[features]
Y = df['Credit Risk?']

dtree = tree.DecisionTreeClassifier(criterion="log_loss")
dtree = dtree.fit(X, Y)
# print(dtree.predict(np.array(df.iloc[12])[1:4].reshape(1, -1)))
print(dtree.predict(np.array([1, 0, 0]).reshape(1, -1)))
print(dtree.predict(np.array([2, 1, 0]).reshape(1, -1)))

tree.plot_tree(dtree, feature_names=features)
plt.show()