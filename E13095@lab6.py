import pandas as pd
from sklearn import tree
from sklearn. model_selection import train_test_split
from sklearn. model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn. naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn. naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('zooData.csv', header=0, sep=',',)
columns = list(df.columns)
columns.remove('type')
columns.remove('animalName')
X = df.as_matrix(columns=columns)
Y =df["type"]

#part one
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print("TrainingAccuracy:" ,clf.score(X,Y))

#part two
X_train , X_test , Y_train , Y_test = train_test_split (X, Y, test_size =0.333 ,random_state =0)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
print("Test Accuracy:" ,clf.score(X_test,Y_test))


#part 3
scores = cross_val_score (clf , X, Y)

#print("10CV Accuracy:)" % (scores.mean (), scores.std ()*2))

# The most suited method for this dataset can be said to be the second method which splits training set for 1/3 and dataset for 2/3.
#this provides enough resources to get a good classification so there is no need to go for a cross vailidation and change resources.



print("confusion matrix")
Y_pred = clf.fit( X_train , Y_train ). predict( X_test )
print(confusion_matrix ( Y_test , Y_pred ))

#naive bayees
clf = GaussianNB () # c l f i s a c l a s s i f i e r
clf.fit( X_train , Y_train )
Y_pred = clf.predict(X_test)
print("confusion matrix")
print(confusion_matrix ( Y_test , Y_pred ))


"""
From the naive bayes matrix we get it predicts 30 correctly(sum of values in diagnols) and 4 incorectly
"""

#cross validation
scores = cross_val_score (clf , X, Y, cv =10)
#print("10CV Accuracy: %0.2f)" % (scores.mean (), scores.std ()*2))

"""CV accuracy for Gaussian Naive Bayes is 0.95 """
#nearest neighbour
clf = neighbors . KNeighborsClassifier ( n_neighbors =1)
clf.fit( X_train , Y_train )
Y_pred = clf.predict(X_test)

print("confusion matrix")
print(confusion_matrix ( Y_test , Y_pred ))

"""
from the matrix we get Nearest neighbour also predicts 30 classes correclty and 4 incorectly
"""

print("Cross Validation")
scores = cross_val_score (clf , X, Y, cv =10)
#print("10CV Accuracy: %0.2f " % (scores.mean (), scores.std ()*2))

"""CV accuracy for  Nearest Neighbor is 0.98 """
#Multinomial Naive Bayes
clf= MultinomialNB ()
clf.fit( X_train , Y_train )

Y_pred = clf.predict(X_test)

print("confusion matrix")
print(confusion_matrix ( Y_test , Y_pred ))

"""
From the confusion matrix for  Multinomial Naive Bayes we can see that it predicts 29 correctly and 5 wrongly
"""


scores = cross_val_score (clf , X, Y, cv =10)
#print("10CV Accuracy: %0.2f " % (scores.mean (), scores.std ()*2))


"""CV accuracy for Multinomial Naive Bayes is 0.89 """


#support vector machine
clf = svm.SVC(kernel="linear", C=1, gamma =1) # c l f i s a c l a s s i f i e r
clf.fit( X_train , Y_train )
Y_pred = clf.predict(X_test)

print("confusion matrix")
print(confusion_matrix ( Y_test , Y_pred ))

"""
From the confusion matrix we get again it predicts 30 correctly and 4 incorectly
"""


scores = cross_val_score (clf , X, Y, cv =10)
#print("10CV Accuracy: %0.2f " % (scores.mean (), scores.std ()*2))

"""CV accuracy for Support Vector Machine is 0.95 """

#MLP for last lab task
clf = MLPClassifier(alpha=1)
clf.fit( X_train , Y_train )
Y_pred = clf.predict(X_test)

print("confusion matrix")
print(confusion_matrix ( Y_test , Y_pred ))

"""
From the confusion matrix we can say that it gets 30 corectly aand 4 incorectly
"""

scores = cross_val_score (clf , X, Y, cv =10)
#print("10CV Accuracy: %0.2f " % (scores.mean (), scores.std ()*2))

"""CV accuracy for MLP Classifier is 0.96 """


