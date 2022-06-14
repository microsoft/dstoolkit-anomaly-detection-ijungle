import os
import argparse
import pandas as pd
from azureml.core import Run, Model
from sklearn import preprocessing
import joblib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, dest='input_')
parser.add_argument("--output", type=str, dest='output_')
parser.add_argument("--feat-id", type=str, dest='feat_id')

run = Run.get_context()

print("Reading parameters ...")
args = parser.parse_args()
output_dir = args.output_
feat_id = args.feat_id
print("Output dir:", output_dir)
print("feat_id:", feat_id)


print("Loading data ...")
df = run.input_datasets['inference_data'].to_pandas_dataframe()
df.set_index(feat_id, inplace=True)
print(df)

print("Loading model ...")
model_name = 'best_iforest.pkl'
model_path = Model.get_model_path(model_name)
model = joblib.load(model_path)
print("model", model)

print(df.isnull().any())

print("Making predictions ...")
y_pred = model.predict(df)
scores = model.score_samples(df)
print("Number of anomalies: ", len(y_pred[y_pred==-1]))

print("Generating outputs ...")
df_out = pd.DataFrame()
df_out[feat_id] = df.index.values
df_out['pred'] = y_pred
df_out['score'] = scores

save_path = os.path.join(output_dir,'results.parquet')
df_out.to_parquet(save_path, index=False)

run.complete()