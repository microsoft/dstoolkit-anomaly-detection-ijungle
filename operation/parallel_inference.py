import os
import numpy as np
from azureml.core import Model
import joblib
#import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument("--id-feature", type=str, dest='id_feature', help='ID Freature')
#args = parser.parse_args()
#id_feat = str(args.id_feature)
#print('id feature',  id_feat)


def init():
    # Runs when the pipeline step is initialized
    global model

    # load the model
    model_path = Model.get_model_path('best_iforest.pkl')
    model = joblib.load(model_path)
    
def run(mini_batch):
    mini_batch.set_index('Van_Stock_Proposal_Detail_Id', inplace=True)
    index_list = list(mini_batch.index)
    y_pred = model.predict(mini_batch).tolist()
    score = model.score_samples(mini_batch).tolist()
    return(list(zip(index_list, y_pred, score)))