import pandas as pd
import numpy as np
import sys
import unittest
from pandas.util.testing import assert_frame_equal


class calculation:
    def __init__(self,Xvector,Yvector):
        self.Xvector=Xvector;
        self.Yvector=Yvector;
        self.xMean = self.Xvector.mean()
        self.yMean = self.Yvector.mean()
        self.n = self.Xvector.size

    def covariance(self):
        finalSum=0
        for i in range(0,self.n):
            xi=self.Xvector[i]
            yi=self.Yvector[i]
            finalSum=finalSum+(((xi-self.xMean)*(yi-self.yMean))/(self.n-1))
        return finalSum

    def correlation(self):
        sdX=self.Xvector.std()
        sdY=self.Yvector.std()
        coValue=self.covariance()
        finalSum=((coValue)/(sdX*sdY))
        return finalSum

df=pd.read_csv('lab03Exercise.csv',names=['a','b','c','d','e'])
df.shape
df=df.fillna(0)

names=['a','b','c','d','e']
matrix=np.zeros(shape=(5,5),dtype=float)
sys.stdout.write("waiting for Covariance Matrix.")
for i in range(0, 5):
    for j in range(0,5):
        if i==j:
            value=df.loc[:,names[i]].var()
        else:
            cal=calculation((df.loc[:,names[i]]),(df.loc[:,names[j]]))
            value =cal.covariance()

        matrix[i][j]=value
        sys.stdout.write('.')

print '\n '
print matrix
print "\n Lets compare the matrix with the library function result\n"
print df.cov()


matrix1=np.zeros(shape=(5,5),dtype=float)
sys.stdout.write("\n Waiting for Correlation Matrix.")
for i in range(0, 5):
    for j in range(0,5):
        cal = calculation((df.loc[:, names[i]]), (df.loc[:, names[j]]))
        value=cal.correlation()
        matrix1[i][j]=value
        sys.stdout.write('.')

print '\n '
print matrix1
print "\n  Lets compare the matrix with the library function result\n"
print df.corr()
unittest.main


class calculationTesting(unittest.TestCase):
    def setUp(self):
        print("caculationTesting: SetUp : Begin")
        self.df = pd.read_csv('lab03Exercise.csv', names=['a', 'b', 'c', 'd', 'e'])
        self.df.shape
        self.df = self.df.fillna(0)

        self.names = ['a', 'b', 'c', 'd', 'e']


    def testCovariance(self):

        matrix = np.zeros(shape=(5, 5), dtype=float)
        for i in range(0, 5):
            for j in range(0, 5):
                if i == j:
                    value = self.df.loc[:, self.names[i]].var()
                else:
                    cal = calculation((self.df.loc[:, self.names[i]]), (self.df.loc[:, self.names[j]]))
                    value = cal.covariance()

                matrix[i][j] = value

        actual = pd.DataFrame(matrix,self.names,self.names)
        correct =self.df.cov()

        assert_frame_equal(actual, correct)
        print "Actual cov :\n", actual, "\n Correct cov: \n", correct

    def testCorrelation(self):

        matrix1 = np.zeros(shape=(5, 5), dtype=float)
        for i in range(0, 5):
            for j in range(0, 5):
                cal = calculation((self.df.loc[:, self.names[i]]), (self.df.loc[:, self.names[j]]))
                value = cal.correlation()
                matrix1[i][j] = value
        actual = pd.DataFrame(matrix1,self.names,self.names)
        correct =self.df.corr()

        assert_frame_equal(actual, correct)
        print "Actual corr :\n", actual, "\n Correct corr: \n", correct

    def tearDown(self):
        print("calculationTesting: tearDown : Begin")
        print("=============================================")


if __name__ == '__main__':
    unittest.main()

#since there is no zero in the covariance matrix we can conclude that all the vectors are corealated







