import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

np.random.seed(7)
data = pd.read_excel('../../data_process/sports.xlsx')
drop_list = ['Label', 'TextID', 'URL']
x = data.drop(drop_list, axis=1)
y = data['Label']

print(x.shape)
print(x.info())
print(x.describe())

plt.rcParams['figure.figsize'] = (15,11)
seaborn.set(rc={'figure.figsize':(15,11)})
seaborn.countplot(x='Label', data=data)

numerical_data = x.drop(['sentence1st', 'sentencelast'], axis=1).columns

print(x['sentence1st'].value_counts())
print(x['sentencelast'].value_counts())

min_max=MinMaxScaler()
x.loc[:, numerical_data] = min_max.fit_transform(x[numerical_data])

feature_variance = x.var().sort_values()
low_var_key = []
threshold = 0.005
for i in range(len(feature_variance.keys())):
    if feature_variance.values[i] <= threshold:
        key = feature_variance.keys()[i]
        low_var_key.append(key)
        print(x[key].value_counts())
    else:
        break
print(low_var_key)
x.drop(low_var_key, axis=1)

corr_series = x.corr(method='pearson').abs().unstack().sort_values(ascending=False)
# print(corr_series[59:100])
high_cor_key=[]
association = x.corr()
colNum = association.shape[0]
for i in range(colNum):
    for j in range(i+1,colNum):
        if association.iloc[i,j]>0.998:
            high_cor_key.append(association.columns[i])
print(high_cor_key)
x.drop(high_cor_key, axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=7, stratify=y)

knn = KNeighborsClassifier()
param_grid = {'n_neighbors': list(range(1, 100, 2)), 'weights': ['uniform', 'distance']}
grid_result = GridSearchCV(estimator=knn, param_grid=param_grid, scoring='roc_auc')
grid_result.fit(x_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

knn = KNeighborsClassifier(n_neighbors=grid_result.best_params_['n_neighbors'], weights=grid_result.best_params_['weights'])
knn.fit(x_train, y_train)
knn_prediction = knn.predict_proba(x_test)
auc = roc_auc_score(y_test, knn_prediction[:, 1])
print(auc)
