# -*- coding: utf-8 -*-
from azureml.core import Model, Run
import argparse
import numpy as np
import iJungle
import pickle

run = Run.get_context()

print("iJungle version:", iJungle.__version__)
run.log('iJungle_version', iJungle.__version__)

parser = argparse.ArgumentParser()

# Input Data
parser.add_argument("--input-data", type=str, dest='input_data', help='Overhead dataset')

# Hyper parameters
parser.add_argument('--trees', type=int, dest='trees', default=100, help='Number of trees')
parser.add_argument('--subsample-size', type=int, dest='subsample_size', default=8192, help='Subsample size')

# Add arguments to args collection
args = parser.parse_args()

# Log Hyperparameter values
trees = np.int(args.trees)
subsample_size = np.int(args.subsample_size)
print('trees',  trees)
print('subsample_size',  subsample_size)
run.log('trees',  trees)
run.log('subsample_size',  subsample_size)

# Load training data
print("Loading Data...")
W = run.input_datasets['overhead_data'].to_pandas_dataframe() # Get the training data from the estimator input

# Load iFor_list pickle
print("Loading pickle...")
model_name = 'iJungle_light_' + str(trees) + '_' + str(subsample_size)
print(model_name)
model_path = Model.get_model_path(model_name)
print(model_path)
with open(model_path, 'rb') as infile:
    iFor_list = pickle.load(infile)


# Evaluation
print("Starting evaluation ...")
os.makedirs(iJungle._MODEL_DIR, exist_ok=True)
results = iJungle.model_eval_fun(W, iFor_list)
results_filename = os.path.join(iJungle._MODEL_DIR, model_name + '_results.pkl')
print("Writing results:", results_filename)
with open(results_filename, 'wb') as outfile:
    pickle.dump(results, outfile)

# Log dummy metric
run.log('Dummy', np.float(0))

run.complete()
