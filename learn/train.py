import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import argparse

# support vector regression
svr = svm.SVR()
# see documentation for possible kernels

# scale features for stochastic grad descent??
# READ ON THIS

# stochastic gradient descent regressor

# decision tree regression
# KNN regression ~~~~ almost certainly inefficient

# gradient tree boosting

# neural nonsense maybe?


# main method
if __name__ == '__main__':
    main()

    parser = argparse.ArgumentParser(
        description='associate threads with all relevant variables')
    parser.add_argument('--name', action='store', type=str)
    parser.add_argument('--dir', action='store', type=str)
    args = parser.parse_args()
    filename = args.name
    subdir = args.dir
