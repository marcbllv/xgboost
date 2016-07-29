import pandas as pd
import numpy as np
import xgboost as xgb

def dummydata():
    '''
        Generate a few data points, with 2 features
    '''
    data = pd.DataFrame(np.array([
        [1, -1, 0],
        [-1, -1, 0],
        [1, 2, 0],
        [-1, 2, 0],

        [-1, 1, 1],
        [-0.8, 1, 1],

        [0.8, 1.2, 2],
        [1.1, 0.6, 2]]),
        columns=['f1', 'f2', 'class'])
    return data.drop('class', axis=1), data['class']

X, y = dummydata()
dtrain = xgb.DMatrix(X, label=y)
dtest  = xgb.DMatrix(X)
#param = {'max_depth':5, 'eta':1, 'silent':1, 'n_estimators': 100, 'objective':'multi:brier', 'num_class':3}
param = {'max_depth':5, 'eta':1, 'silent':1, 'objective':'multi:brier', 'num_class':3}
n_trees = 100
print "--- DATA ---"
print X
print y
print "--- PARAMS ---"
print param
print "%d trees" % n_trees 
print "--- training starts ---"

bst = xgb.train(param, dtrain, num_boost_round=n_trees)
print "--- PREDICTION ---"
print bst.predict(dtest)
