from azureml.core import Run
import argparse
import pandas as pd
import os
import iJungle

run = Run.get_context()
parser = argparse.ArgumentParser()

# Input Data
parser.add_argument("--input-data", type=str, dest='prepped_data', help='Prepped data')
parser.add_argument('--overhead-data', type=str, dest='overhead_data', help='Overhead data')
parser.add_argument('--overhead-expected-m', type=str, dest='overhead_expected_m')

# Add arguments to args collection
args = parser.parse_args()
prepped_data = args.prepped_data
print("Prepped folder", prepped_data)
overhead_data = args.overhead_data
print("Model input", overhead_data)
overhead_expected_m = int(args.overhead_expected_m)
print("overhead_expected_m", overhead_expected_m)

# Load training data
print("Loading Data...")
load_path = os.path.join(prepped_data,'prepped.parquet')
df = pd.read_parquet(load_path)
print("Data loaded. Shape:", df.shape)

# Overhead sample size calculation
n_records = df.shape[0]
overhead_size = min(1,overhead_expected_m/n_records)
print("Overhead size", overhead_size)
run.log('overhead_size',  overhead_size)

W = iJungle.select_overhead_data(df, overhead_size=overhead_size)
print("Overhead shape", W.shape)

print("Saving Data...")
save_path = os.path.join(overhead_data,'W.parquet')
W.to_parquet(save_path, index=False)

run.complete()