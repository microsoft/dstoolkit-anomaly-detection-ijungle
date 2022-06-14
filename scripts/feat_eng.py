import os
import argparse
import pandas as pd
from azureml.core import Run, Model
from sklearn import preprocessing
import joblib
import numpy as np

LOCAL_MODEL_PATH = 'outputs'

parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, dest='train_dataset_id')
parser.add_argument('--prepped-data', type=str, dest='prepped_data')
parser.add_argument('--index-feature', type=str, dest='index_feature')
parser.add_argument('--training', type=str, dest='training')

print('Loading parameters ...')
args = parser.parse_args()
save_folder = args.prepped_data
index_feature = args.index_feature
training = bool(args.training)

print('save_folder', save_folder)
print('index_feature', index_feature)
print('training', training)

run = Run.get_context()

print("Loading data ...")
df = run.input_datasets['input'].to_pandas_dataframe()
print(df)
print("Shape of data:",df.shape)

print("Setting index ...")
df.set_index(index_feature, inplace=True)

for feat in df.columns:
    if training:
        print("Training and registering scaler for feature:", feat)
        scaler = preprocessing.StandardScaler()
        scaler.fit(df[[feat]])
        model_name = 'ijungle_scaler_model_'+feat
        file_name = os.path.join(LOCAL_MODEL_PATH, model_name + '.pkl')
        joblib.dump(value=scaler, filename=file_name)
        Model.register(
            workspace=run.experiment.workspace,
            model_path = file_name,
            model_name = model_name
        )
    else:
        print("Applying scaler for feature:", feat)
        model_name = 'invoices_scaler_model_'+feat
        model_path = Model.get_model_path(model_name)
        scaler = joblib.load(model_path)        
    df[feat] = scaler.transform(df[[feat]]).reshape(df.shape[0])

print("Reseting index ...")
df.reset_index(inplace=True)

print("Saving Data...")
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder,'prepped.parquet')
df.to_parquet(save_path, index=False)

run.complete()