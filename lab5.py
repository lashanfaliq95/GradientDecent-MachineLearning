from sklearn import tree
import pandas as pd
import numpy as np
import sklearn
import pydotplus


df=pd.read_csv('breaset-cancer.csv',names=['country','income per person','alc consumption','armed force rate','breast cancer per 100th person','co2 emmision','female employee rate','hivrate','internet use rate','lifeexpentancy','oil per person','polity score','Relectric per person','suicide 100th person','employee rate','urban rate'])
df.shape
df=df.fillna(0)
df= df.ix[1:] #removing first row
df['breast cancer per 100th person']=np.where(df['breast cancer per 100th person']>20,1,0) #if vale more than 20 one else zero

trainingset=df.iloc[:(len(df.index)/3)*2] #getting 2/3 as training set
trainingtarget=trainingset['country']  #training set target
trainingset=trainingset.drop('country', axis=1)
testset=df.iloc[(len(df.index)/3)*2:] #getting balnce as test set
testsettrue=testset['country'] #real predictions for traingn set
testset=testset.drop('country', axis=1)

clf = tree. DecisionTreeClassifier()
clf = clf.fit(trainingset,trainingtarget) #using the training set to create a tree
predicted=clf.predict(testset) #predicting countirs for test set

print (sklearn.metrics.accuracy_score(testsettrue,predicted))
