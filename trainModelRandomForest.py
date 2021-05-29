import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score,cohen_kappa_score,f1_score,matthews_corrcoef,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from urllib.parse import urlparse
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#can be called directly or as a module from another script
if __name__ == "__main__" or __name__== "trainModelRandomForest":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    # Create a Gaussian Classifier
    from MLFlowRun import n_estimators
    from PreprocessPinguins import X_train_fit, X_test_fit, y_train
    trainedRF = RandomForestClassifier(n_estimators=n_estimators, random_state=1234)

    # Train the model using the training set
    trainedRF.fit(X_train_fit, y_train)
    # predict
    y_pred = trainedRF.predict(X_test_fit)

    print("DONE with training")

