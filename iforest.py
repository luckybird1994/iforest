import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(10)

X = (.7 *rng.randn(100, 8))+3
X_outliers = rng.uniform(low=1, high=9, size=(20, 8))

X_train = np.concatenate((X,X_outliers),axis=0)
print( X_train.shape )

X_testoutliers = rng.uniform(low=1, high=9, size=(10, 8))
X_testnorm = (.7 *rng.randn(10, 8))+3
X_test = np.concatenate((X_testnorm,X_testoutliers),axis=0)
print( X_test.shape )

isofortrain = IsolationForest(n_estimators = 1000,
                             max_samples = 'auto',
                             contamination = .20,
                             max_features = 1,
                             random_state = rng,
                             n_jobs = -1)

isofortrain.fit(X_train)
anomalytrain = isofortrain.decision_function(X_train)
predicttrain = isofortrain.predict(X_train)

print( predicttrain )
testframe = pd.DataFrame(X_train)
print( testframe.shape )

testframe['score'] = anomalytrain
testframe['outlier'] = predicttrain
print( testframe.tail() )