import os
import numpy as np
from azureml.core import Model
import joblib


def init():
    # Runs when the pipeline step is initialized
    global model

    # load the model
    model_path = Model.get_model_path('best_iforest.pkl')
    model = joblib.load(model_path)
    
def run(mini_batch):
#     # This runs for each batch
#     resultList = model.predict(mini_batch)
#     ind = mini_batch.index
#     return resultList.tolist()
    index_list = list(mini_batch.index)
    y_pred = model.predict(mini_batch).tolist()
    score = model.score_samples(mini_batch).tolist()
    
    
    return(list(zip(index_list, y_pred, score)))