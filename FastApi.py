from typing import List

from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import HTMLResponse
import mlflow.sklearn

from sklearn.metrics import classification_report, accuracy_score,cohen_kappa_score,f1_score,matthews_corrcoef,precision_score,recall_score
import os
import json
from pandas import read_csv,get_dummies,DataFrame
import io
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

app = FastAPI()

# load model
def load_model_utilities(best_model_hash):
    #find script and mlflow location
    whereAmI = os.getcwd()
    #load model
    model = mlflow.sklearn.load_model(whereAmI + '/mlruns/0/'+ best_model_hash + '/artifacts/RandomForestClassifier')
    #model = mlflow.sklearn.load_model(whereAmI + '/mlruns/0/' + '6711c94fe3134484aeb4eaedbc091084' + '/artifacts/RandomForestClassifier')
    # load features used
    with open(whereAmI + '/mlruns/0/'+ best_model_hash + '/artifacts/RFfeatures.txt',
            "r") as f:
        modelfeatures = eval(f.readline())
        f.close()
    #load the predicted target
    with open(whereAmI + '/mlruns/0/'+ best_model_hash + '/artifacts/target.txt',
              "r") as t:
        target = t.readline()
        t.close()
    return model, modelfeatures, target


@app.get("/")
# this functon defines the landing page of the FastAPI tool
async def main():
    content = """
<body>
  <header>
    <h1>Welcome to MLFlow automatizer</h1>
    <p>Check your model by submitting a .csv file with known classes</p>
    <p>Please use .csv</p>
  </header>
<label for="myfile">Select a file:</label>
<form action="/classified_input/" enctype="multipart/form-data" method="post" >
<input name="file" type="file" Content-Type= 'multipart/form-data'>
<input type="submit">
</form>

<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)

def readTxt(file): #
    return file.read()

def turn_to_json(df): #function turns pandas df to json for display
    json_format = df.to_json(orient='records')
    json_format = json.loads(json_format)
    return json_format

def turn_to_pandasDF(file): #function turns file from html form to pandas df
    response = file.decode()
    binaryStream = io.BytesIO(response.encode())
    df = read_csv(binaryStream)
    # drop NA
    df = df.dropna()
    return df

def input_cleaning(pd_dataframe, required_features): #function cleans up test dataframe
    # drop NA
    df = pd_dataframe.dropna()
    # transform categorical columns get categoricals
    cols_to_transform = df.select_dtypes(include=['object', 'category']).columns
    # encode categoricals to numeric
    df_hotone = get_dummies(data=df, columns=cols_to_transform, drop_first=False)
    # extract only those columns needed for predictions
    test = df_hotone[required_features]
    return test

def determine_best_model(): #function goes through MLflow models and picks the most accurate one
    run = MlflowClient().search_runs(
        experiment_ids="0",
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.Accuracy DESC"]
    )[0]
    best_model_id = str(run.info.run_id)
    return best_model_id
#
# def classify_testdata_return_labels(model, testframe):  # function classifies a test dataframe and returns labels only
#     label = model.predict(testframe)
#     return label

def classify_testdataframe(model, testset): #function predicts probabilities for provided testset (pandas df) for each class and returns a dataframe
    label = model.predict(testset)
    res_prob = model.predict_proba(testset)
    modelclasses = model.classes_
    #build up dataframe
    df = DataFrame({'speciespredicted': label})
    for i in range(modelclasses.size):
        df[modelclasses[i] + '_Probability'] = res_prob[:, i]
    return label, df
def eval_metrics_classification(actual, pred): #function returns classification metrics
    f1 = f1_score(actual, pred, average='micro')
    acc = accuracy_score(actual, pred)
    ck = cohen_kappa_score(actual, pred)
    mcc = matthews_corrcoef(actual, pred)
    prec = precision_score(actual, pred, average='micro')
    recall = recall_score(actual, pred, average='micro')
    return acc, prec, recall, f1, ck, mcc

@app.post("/classified_input/")
async def classify_upload_file(file: UploadFile = File(...)):
    text_binary = await readTxt(file) #the await is crucial, also for some reason the wrapping
    #convert File to pandas df
    df = turn_to_pandasDF(text_binary)
    #get head for display
    df_head = df.head(5)
    head = turn_to_json(df_head)
    # get best model from mlflow dir, as well as model's features
    best_model = determine_best_model()
    model, modelfeatures, target = load_model_utilities(best_model)
    # clean and transform input
    test = input_cleaning(df, modelfeatures)
    #prediction
    response, prediction_probas = classify_testdataframe(model, test)

    #TODO revert to "in", once you finished working on else part
    if target not in df.columns: #if target column is in dataframe, then a model assessmemnt is done, else only prediction returned
        class_report = classification_report(df[target], response)
        (acc, prec, recall, f1, ck, mcc) = eval_metrics_classification(df[target], response)

        if acc > 0.5: #TODO: insert condition to trigger mlflow
            tester = "acc > 0.5"
      #      os.system('python /home/elena/PycharmProjects/FUH_MLFlow/trainModelRandomForest.py 50')
        else:
            tester = "acc < 0.5"
        return {
            "head of input": head,
            "target": target,
            "confusion matrix": class_report,
            "acc": acc,
            "prec": prec,
            "recall": recall,
            "f1": f1,
            "ck": ck,
            "mcc": mcc
        }
    else: #the provided df did not contain a target to test the model against
        label,  prediction_df = classify_testdataframe(model, test)
        label = str(label)
        #prediction_df_head = prediction_df.head(5)
        #prediction_df_head = turn_to_json(prediction_df_head)
        return {
            "head of input": label
        }

#from here on starts experimental part. Delete, once done

@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    return {"filenames": [file.filename for file in files]}

# https://towardsdatascience.com/how-you-can-quickly-deploy-your-ml-models-with-fastapi-9428085a87bf

# https://testdriven.io/blog/fastapi-machine-learning/
testsample: list[float] = [52.7, 19.8, 197.0, 3725.0, 1.0, 0.0, 1.0]
@app.get("/predict")
def get_prediction():
    #lbl, probs = classify_sample(model, testsample)
    # response_object = {"Class": lbl, "PROB":prob}
    # response_object = {"Predicted class": lbl}
    # get best model from mlflow dir, as well as model's features
    best_model = determine_best_model()
    model, modelfeatures, target = load_model_utilities(best_model)
    if True:
        response_object = list(show_listloop(model, testsample))
        #os.system('python /home/elena/PycharmProjects/FUH_MLFlow/trainModelRandomForest.py 50')
        return response_object
    else:
        return {"Nope"}



def classify_sample(model, sample):  # function classifies a single sample based on provided model
    label = model.predict([sample])[0]
    res_prob = model.predict_proba([sample])
    # return {'label': label, 'class_probability': res_prob}
    return label, res_prob



def show_listloop(model, sample):  # function prints class probabilities of every class predicted by the model
    res_prob = model.predict_proba([sample])
    modelclasses = model.classes_
    # Create a generator
    for i in range(modelclasses.size):
        yield (modelclasses[i], "Probability: ", res_prob[0][i])
