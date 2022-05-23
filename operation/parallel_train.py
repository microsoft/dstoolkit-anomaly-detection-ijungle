# -*- coding: utf-8 -*-
from azureml.core import Run
import argparse
import numpy as np
import iJungle

run = Run.get_context()
print("iJungle version:", iJungle.__version__)
run.log('iJungle_version', iJungle.__version__)

parser = argparse.ArgumentParser()

# Input Data
parser.add_argument("--input-data", type=str, dest='input_data', help='training dataset')
parser.add_argument("--id-feature", type=str, dest='id_feature', help='ID Freature')
parser.add_argument("--max-subsample-size", type=int, dest='max_sss', help='Max subsample size')
parser.add_argument("--train-size", type=float, dest='train_size', help='Train size')

# Hyper parameters
parser.add_argument('--trees', type=int, dest='trees', default=100, help='Number of trees')
parser.add_argument('--subsample-size', type=int, dest='subsample_size', default=8192, help='Subsample size')

# Add arguments to args collection
args = parser.parse_args()
id_feat = str(args.id_feature)
print('id feature',  id_feat)

# Log Hyperparameter values
trees = np.int(args.trees)
subsample_size = np.int(args.subsample_size)
print('trees',  trees)
print('subsample_size',  subsample_size)
run.log('trees',  trees)
run.log('subsample_size',  subsample_size)

# Other parameters
max_sss = np.int(args.max_sss)
train_size = np.float(args.train_size)
print("Max subsample size", max_sss)
print("Train size", train_size)
run.log('max_sss',  max_sss)
run.log('train_size',  train_size)

# Load training data
print("Loading Data...")
df = run.input_datasets['training_data'].to_pandas_dataframe() # Get the training data from the estimator input
df.set_index(id_feat, inplace=True)

print("Starting training ...")
model_filename = iJungle.model_train_fun(df, trees, subsample_size, train_size, max_sss)
print(model_filename)

# Log dummy metric
run.log('Dummy', np.float(0))

run.complete()
