import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df1=pd.read_csv('dataset/lab04ExerciseAngles.csv',names=['angle1','angle2','angle3'])
df1=df1.fillna(df1.mean())

df2=pd.read_csv('dataset/lab04ExerciseChannels.csv',names=['channel1','channel2','channel3','channel4','channel5'])
df2=df2.fillna(df2.mean())

mergeddf = pd.concat([df1, df2], axis=1)

#we will have to use linear regression since the outcome is not binary but a value of an angle.

#angle one isnt a good choice because it doesnt vary much throught
#angle two and three show good variation so i wil choose angle 3
#channel one and channel4 shows good variation with the chosen angle column

#so my final choices are angle 3 and channel1,channel4

X=mergeddf[['channel1','channel4']]
Y=mergeddf['angle3']
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)
predicted=lin_reg.predict(X_val)

# The mean squared error
print("Mean squared error: %.2f"
%mean_squared_error(y_val, predicted))

print('Variance score: %.2f' % r2_score(y_val, predicted))

#i would have to say the accuracy is very low