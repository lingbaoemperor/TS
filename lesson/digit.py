from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

#Forest
path_train = './digit/train.csv'
path_test = './digit/test.csv'
train = pd.read_csv(path_train)
test = pd.read_csv(path_test)
train = np.array(train)
test = np.array(test)
y = train[:,0]
X = train[:,1:]
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X,y)
result=clf.predict(test)
print(result)
#cross_val_score(clf, iris.data, iris.target, cv=10)
index = np.arange(1,28001)
result = np.c_[index,result]
res = pd.DataFrame(result,columns=['ImageId','Label'])
res.to_csv('./digit/result.csv',index=False)
#print(neigh.predict_proba([[0.9]])


#KNN
#path_train = './digit/train.csv'
#path_test = './digit/test.csv'
#train = pd.read_csv(path_train)
#test = pd.read_csv(path_test)
#train = np.array(train)
#y = train[:,0]
#X = train[:,1:]
#test = np.array(test)
#neigh = KNeighborsClassifier(n_neighbors=9)
#neigh.fit(X, y)
#result = neigh.predict(test)
#index = np.arange(1,28001)
#result = np.c_[index,result]
#res = pd.DataFrame(result,columns=['ImageId','Label'])
#res.to_csv('./digit/result.csv',index=False)
##print(neigh.predict_proba([[0.9]]))