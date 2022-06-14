from azureml.core import Run, Model
import argparse
import pandas as pd
import numpy as np
import joblib
import os
import iJungle
import shutil

run = Run.get_context()
parser = argparse.ArgumentParser()

# Input Data
parser.add_argument("--training-folder", type=str, dest='training_folder', help='training data folder')
parser.add_argument("--max-subsample-size", type=int, dest='max_sss', help='Max subsample size')
parser.add_argument("--model-output", type=str, dest='model_output', help='model output folder')
parser.add_argument("--id-feat", type=str, dest='id_feat')
parser.add_argument("--train-expected-m", type=str, dest='train_expected_m')

# Hyper parameters
parser.add_argument('--trees', type=int, dest='trees', default=100, help='Number of trees')
parser.add_argument('--subsample-size', type=int, dest='subsample_size', default=8192, help='Subsample size')


# Add arguments to args collection
args = parser.parse_args()
training_folder = args.training_folder
print("Training folder", training_folder)
model_output = args.model_output
print("Model output", model_output)
id_feat = args.id_feat
print("id_feat", id_feat)
train_expected_m = int(args.train_expected_m)
print("train_expected_m", train_expected_m)

# Log Hyperparameter values
trees = np.int(args.trees)
subsample_size = np.int(args.subsample_size)
print('trees',  trees)
print('subsample_size',  subsample_size)
run.log('trees',  trees)
run.log('subsample_size',  subsample_size)

# Other parameters
max_sss = np.int(args.max_sss)
print("Max subsample size", max_sss)
run.log('max_sss',  max_sss)

# Load training data
print("Loading Data...")
load_path = os.path.join(training_folder,'prepped.parquet')
df = pd.read_parquet(load_path)
df.set_index(id_feat, inplace=True)

# Train sample size calculation
n_records = df.shape[0]
train_size = min(1,train_expected_m/n_records)
print("Train size", train_size)
run.log('train_size',  train_size)

print("Starting training ...")
model_filename = iJungle.model_train_fun(df, trees, subsample_size, train_size, max_sss)
print(model_filename)
model_name = 'iJungle_light_' + str(trees) + '_' + str(subsample_size)
print(model_name)

model_path = os.path.join(iJungle._MODEL_DIR, model_filename)

print('Registering model...')
Model.register(
    workspace=run.experiment.workspace,
    model_path = model_path,
    model_name = model_name,
    properties={
        'trees':trees,
        'subsample_size':subsample_size})

# Log dummy metric
run.log('Dummy', np.float(0))

shutil.copy(model_path, os.path.join(model_output,model_name + '.pkl'))

run.complete()