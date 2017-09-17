from sklearn import tree
import pandas as pd
import numpy as np
import sklearn
import pydotplus

def string_to_float(X_train,column):
	"""to convert string columns into float"""
	S = set(X_train[:,column]) # collect unique label names
	D = dict( zip(S, range(len(S))) ) # assign each string an integer, and put it in a dict
	X_train[:,column] = [D[x] for x in X_train[:,column]] # store class labels as ints


df=pd.read_csv('breaset-cancer.csv',names=['country','income per person','alc consumption','armed force rate','breast cancer per 100th person','co2 emmision','female employee rate','hivrate','internet use rate','lifeexpentancy','oil per person','polity score','Relectric per person','suicide 100th person','employee rate','urban rate'])
df.shape
df=df.fillna(0)
df= df.ix[1:] #removing first row
df['breast cancer per 100th person']=np.where(df['breast cancer per 100th person']>20,1,0) #if vale more than 20 one else zero
trainingset=df.iloc[:(len(df.index)/3)*2] #getting 2/3 as training set
trainingtarget=trainingset['breast cancer per 100th person']  #training set target
trainingset=trainingset.drop('breast cancer per 100th person', axis=1)
testset=df.iloc[(len(df.index)/3)*2:] #getting balnce as test set
testsettrue=testset['breast cancer per 100th person'] #real predictions for traingn set
testset=testset.drop('breast cancer per 100th person', axis=1)

k = list(df.columns) #converting the name column to float
X = trainingset.as_matrix(columns=k)
string_to_float(X,0)
a = np.matrix(X)
train=pd.DataFrame(a)



Y= testset.as_matrix(columns=k) #converting the name column to float
string_to_float(Y,0)
b = np.matrix(Y)
test=pd.DataFrame(b)



clf = tree. DecisionTreeClassifier()
clf = clf.fit(train,trainingtarget) #using the training set to create a tree
predicted=clf.predict(test) #predicting breast cancer for test set

print (sklearn.metrics.accuracy_score(testsettrue,predicted))
