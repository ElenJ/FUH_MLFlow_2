import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score,cohen_kappa_score,f1_score,matthews_corrcoef,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from urllib.parse import urlparse
from sklearn.impute import SimpleImputer
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
#rrr
def determine_best_model():
    run = MlflowClient().search_runs(
        experiment_ids="0",
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.Accuracy DESC"]
    )[0]
    best_model_id = str(run.info.run_id)
    return best_model_id
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
test = penguins.drop(columns=[target])  # Features

# load model
best_model = determine_best_model()
model = mlflow.sklearn.load_model('/home/elena/PycharmProjects/FUH_MLFlow/mlruns/0/'+ best_model + '/artifacts/RandomForestClassifier')

#now I want to access ml flow directly
#model_name = "RandomForestClassifier"
#model_version = 1

#model = mlflow.pyfunc.load_model(
#    model_uri=f"models:/{model_name}/{model_version}"
#)
run = MlflowClient().search_runs(
    experiment_ids="0",
    filter_string="",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=1,
    order_by=["metrics.Accuracy DESC"]
)[0]
#model.predict(data)

if __name__ == "__main__":
    prediction = model.predict(test)
    print(prediction)
    # print(run)
    # print(type(run))
    # print("run_id:{}".format(run.info.run_id)) #way to access run_id
    # print(type(str(run.info.run_id)))
    # print((str(run.info.run_id))) #and here as string
    # print("--")