from azureml.core import Run, Model
import argparse
import pandas as pd
import os
import iJungle
import joblib

run = Run.get_context()
parser = argparse.ArgumentParser()

# Input Data
parser.add_argument('--overhead-input', type=str, dest='overhead_input', help='Overhead input')
parser.add_argument('--subsample-list', type=str, dest='subsample_list')
parser.add_argument('--trees-list', type=str, dest='trees_list')

# Add arguments to args collection
args = parser.parse_args()
overhead_input = args.overhead_input
print("Overhead input", overhead_input)
subsample_list = eval(args.subsample_list)
print("subsample_list", subsample_list)
trees_list = eval(args.trees_list)
print("subsample_list", trees_list)

# Load models
print("Loading Models...")
results_dic = {}
for subsample_size in subsample_list:
    results_dic[str(subsample_size)] = {}
    for trees in trees_list:
        model_name = 'iJungle_light_' + str(trees) + '_' + str(subsample_size) + '_results'
        print(model_name)
        model_path = Model.get_model_path(model_name)
        print(model_path)
        results_dic[str(subsample_size)][str(trees)] = joblib.load(model_path)

results = pd.DataFrame(results_dic)

# Calculating best iForest
print("Calculating best iForest ...")
best_subsample_size, best_trees, best_iF_k = iJungle.best_iforest_params(results)

model_name = 'iJungle_light_' + str(trees) + '_' + str(subsample_size)
print("Best iForest model name:", model_name)
model_path = Model.get_model_path(model_name)
print("Loading best iFor_list from ", model_path)
iFor_list = joblib.load(model_path)
model = iFor_list[best_iF_k]
print("Model selected!")
print("Registering model...")
best_model_name = 'best_iforest.pkl'
best_model_path = os.path.join(iJungle._MODEL_DIR, best_model_name)
joblib.dump(model, best_model_path)
Model.register(workspace=run.experiment.workspace ,model_path=best_model_path, model_name=best_model_name)

run.complete()