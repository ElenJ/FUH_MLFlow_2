import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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

