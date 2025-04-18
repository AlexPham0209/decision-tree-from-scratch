import os
import pandas
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

PATH = os.path.join('data', 'credit.csv')
df = pandas.read_csv(PATH)

# Convert categorical data to discrete numerical
df['Credit History'] = df['Credit History'].map({'good': 0, 'bad': 1, 'unknown': 2})
df['Debt'] = df['Debt'].map({'low': 0, 'high': 1})
df['Collateral'] = df['Collateral'].map({'adequate': 0, 'none': 1})
df['Credit Risk?'] = df['Credit Risk?'].map({'low': 0, 'high': 1})

features = ['Credit History', 'Debt', 'Collateral']
X = df[features]
Y = df['Credit Risk?']

dtree = tree.DecisionTreeClassifier(criterion="log_loss")
dtree = dtree.fit(X, Y)
# print(dtree.predict(np.array(df.iloc[12])[1:4].reshape(1, -1)))
print(dtree.predict(np.array([0, 1, 0]).reshape(1, -1)))
print(dtree.predict(np.array([2, 0, 0]).reshape(1, -1)))

tree.plot_tree(dtree, feature_names=features)
plt.show()