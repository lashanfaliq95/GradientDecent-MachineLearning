from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
print iris

X = iris["data"][:, 3:]
Y = (iris["target"] == 2).astype(np.int)
# To train a logistic regression model


X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_proba = log_reg.predict(X_val)

# To train a linear regression model



X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg.predict(X_val)
