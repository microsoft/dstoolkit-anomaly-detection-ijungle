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
parser.add_argument("--overhead-folder", type=str, dest='overhead_folder', help='overhead data folder')
parser.add_argument("--model-input", type=str, dest='model_input', help='model input folder')
parser.add_argument("--overhead-output", type=str, dest='overhead_output', help='overhead output folder')
parser.add_argument("--id-feat", type=str, dest='id_feat')

# Hyper parameters
parser.add_argument('--trees', type=int, dest='trees', default=100, help='Number of trees')
parser.add_argument('--subsample-size', type=int, dest='subsample_size', default=8192, help='Subsample size')


# Add arguments to args collection
args = parser.parse_args()
overhead_folder = args.overhead_folder
print("Overhead folder", overhead_folder)
model_input = args.model_input
print("Model input", model_input)
overhead_output = args.overhead_output
print("Overhead output", overhead_output)

# Log Hyperparameter values
trees = np.int(args.trees)
subsample_size = np.int(args.subsample_size)
print('trees',  trees)
print('subsample_size',  subsample_size)
run.log('trees',  trees)
run.log('subsample_size',  subsample_size)

# Other parameters
id_feat = args.id_feat
print("id_feat", id_feat)
run.log('id_feat',  id_feat)

# Load training data
print("Loading Data...")
load_path = os.path.join(overhead_folder,'W.parquet')
W = pd.read_parquet(load_path)
W.set_index(id_feat, inplace=True)
print("Overhead Data loaded. Shape:", W.shape)


# Load iFor_list pickle
print("Loading pickle...")
model_name = 'iJungle_light_' + str(trees) + '_' + str(subsample_size)
print(model_name)
model_path = Model.get_model_path(model_name)
print(model_path)
iFor_list = joblib.load(model_path)

# Evaluation
print("Starting evaluation ...")
os.makedirs(iJungle._MODEL_DIR, exist_ok=True)
results = iJungle.model_eval_fun(W, iFor_list)
results_filename = os.path.join(iJungle._MODEL_DIR, model_name + '_results.pkl')
print("Writing results:", results_filename)
joblib.dump(value=results, filename=results_filename)

model_name = 'iJungle_light_' + str(trees) + '_' + str(subsample_size) + '_results'
print(model_name)

print('Registering model...')
Model.register(
    workspace=run.experiment.workspace,
    model_path = results_filename,
    model_name = model_name,
    properties={
        'trees':trees,
        'subsample_size':subsample_size})

# Log dummy metric
run.log('Dummy', np.float(0))

shutil.copy(results_filename, os.path.join(overhead_output, model_name + '.pkl'))

run.complete()