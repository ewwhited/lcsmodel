import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('LCS_Clean.csv', delimiter=',', skiprows=1, usecols=(70,71))
Y = np.loadtxt('LCS_Clean.csv', delimiter=',', skiprows=1, usecols=(16))
plt.scatter(data[:,0], data[:,1], c=Y)
plt.xlabel('Gold Difference at 15')
plt.ylabel('xp Difference at 15')
#plt.show()

import scipy.stats as sp
X = np.hstack( (np.ones((2640, 1)), data[:,:]))
beta = np.linalg.inv(X.T@X)@X.T@Y
Ymean = np.mean(Y)
num = ( (X@beta).T@(X@beta)) - 2640*Ymean*Ymean
den = (3-1)*(1/(2640-3)*((Y-X@beta).T@(Y-X@beta)))
ANOVAT = num/den
ANOVAC = sp.f.ppf(.95, 3-1, 2640-3)
ANOVApval = 1-sp.f.cdf(ANOVAT, 3-1, 2640,3)

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LogisticRegression(solver='newton-cg', penalty='none')
model.fit(data, Y)
#print(model.coef_[0], model.intercept_)

model2 = LinearDiscriminantAnalysis()
model2.fit(data, Y)
#print(model2.coef_)

#print(model.score(data, Y), model2.score(data, Y))

from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
Xs, Ys = shuffle(data, Y)
#print(cross_val_score(model, Xs, Ys, cv=5).mean(), cross_val_score(model2, Xs, Ys, cv=5).mean())

prediction = np.loadtxt('data.csv', delimiter=',', usecols=(1, 2))
comparison = np.loadtxt('data.csv', delimiter=',', usecols=(0))
pred = model.predict(prediction)
preds = np.expand_dims(pred, axis=1)

diff = np.absolute(pred-comparison)
#print(np.mean(diff)*69409)
#np.savetxt("prediction.csv", preds, delimiter=",", fmt='%i')
