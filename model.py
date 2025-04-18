import os
import pandas
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

PATH = os.path.join('data', 'credit.csv')
df = pandas.read_csv(PATH)

df['Credit History'] = df['Credit History'].astype('category').cat.codes
df['Debt'] = df['Debt'].astype('category').cat.codes
df['Collateral'] = df['Collateral'].astype('category').cat.codes
df['Credit Risk?'] = df['Credit Risk?'].astype('category').cat.codes

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