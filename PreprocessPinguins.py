
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


#can be called directly or as a module from another script
if __name__ == "__main__" or __name__ == "PreprocessPinguins" :
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the penguin csv file from disc
    penguins = pd.read_csv(
        "/media/elena/5A2D54AF7BBEB818/EigeneDateien/Fernuni Informatik/_Praktikum/Programming/data/penguins_size.csv")

    #Data preprocessing TODO wrap in functions
    # I would replace '.', but not sure how to implement this automatically
    penguins.loc[penguins['sex'] == '.', 'sex'] = np.nan

    # Fill NA by setting strategy to 'most frequent' to impute by the mean
    imputer = SimpleImputer(strategy='most_frequent')  # strategy can also be mean or median
    penguins.iloc[:, :] = imputer.fit_transform(penguins)

    #Hot-One encoding of categoricals
    # which columns are categorical?
    target = 'species'
    cols_to_transform = penguins.select_dtypes(include=['object', 'category']).columns
    # drop target column
    cols_to_transform = cols_to_transform.drop(target)
    # encode categoricals to numeric
    penguins = pd.get_dummies(data=penguins, columns=cols_to_transform, drop_first=True)

    #separate features from target
    X = penguins.drop(columns=[target])  # Features
    y = penguins[target]  # Labels


    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test

    # store used features
    feature_names = str(list(X_train.columns))

    #scale
    sc = StandardScaler()
    X_train_fit = sc.fit_transform(X_train)
    X_test_fit = sc.transform(X_test)

    print("DONE with preprocessing")


