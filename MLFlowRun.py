import warnings
import sys

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score,cohen_kappa_score,f1_score,matthews_corrcoef,precision_score,recall_score

from urllib.parse import urlparse
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics_regression(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def eval_metrics_classification(actual, pred):
    f1 = f1_score(actual, pred, average='micro')
    acc = accuracy_score(actual, pred)
    ck = cohen_kappa_score(actual, pred)
    mcc = matthews_corrcoef(actual, pred)
    prec = precision_score(actual, pred, average='micro')
    recall = recall_score(actual, pred,  average='micro')
    return acc, prec, recall, f1, ck, mcc

# predefine hyperparameters by user's input
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    #import PreprocessPinguins #this module contains all precursor steps for getting X_train, y_train from Penguins file
    from PreprocessPinguins import y_test,feature_names,target

    with mlflow.start_run(): #start of mlflow tracking

        from trainModelRandomForest import trainedRF, y_pred #this module contains the trained Model and the predictions
        #evaluate
        (acc, prec, recall, f1, ck, mcc) = eval_metrics_classification(y_test, y_pred)

        print("  Randomforest model (number of trees=%f):" % n_estimators)
        print("  Accuracy: %s" % acc)
        print("  Precision: %s" % prec)
        print("  Recall: %s" % recall)
        print("  F1: %s" % f1)
        print("  Cohen's Kappa: %s" % ck)
        print("  Matthews correlation coefficient (MCC): %s" % mcc)

        mlflow.log_param("Number of trees", n_estimators)
        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("Precision", prec)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("Cohens Kappa", ck)
        mlflow.log_metric("MCC", mcc)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Log an artifact (output file)
        with open("RFfeatures.txt", "w") as f:
            f.write(str(feature_names))
        mlflow.log_artifact("RFfeatures.txt")
        with open("target.txt", "w") as f:
            f.write(str(target))
        mlflow.log_artifact("target.txt")

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            print("condition was met")
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(sk_model=trainedRF, artifact_path="model",
                                     registered_model_name="RandomForestClassifier")
            # create registered model
            client = MlflowClient()
            client.create_registered_model("RandomForestClassifier")
        else:
            print("else condition")
            mlflow.sklearn.log_model(trainedRF, "RandomForestClassifier")
            # mlflow.sklearn.log_model(sk_model=clf, artifact_path="model", registered_model_name="RandomForestClassifier")
            # create registered model
            # client = MlflowClient()
            # client.create_registered_model("RandomForestClassifier")
            # mlflow.register_model(
            #     "runs:/5a5749d5907440fcbd63d48dc7a3da77/model",
            #     "RandomForestClassifier"
            # )
