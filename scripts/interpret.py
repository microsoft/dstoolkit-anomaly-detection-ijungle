import os
import argparse
import pandas as pd
from azureml.core import Run, Model
from sklearn import preprocessing
import joblib
import numpy as np
from interpret.ext.blackbox import TabularExplainer
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, dest='input', help='inference dataset')
parser.add_argument("--dataprep", type=str, dest='dataprep', help='dataset')
parser.add_argument('--output', type=str, dest='output_dir', help='Folder for results')
parser.add_argument('--index-id', type=str, dest='index_id')
parser.add_argument('--anomaly-score', type=str, dest='anomaly_score')

run = Run.get_context()

print("Loading arguments ...")
args = parser.parse_args()
output_dir = args.output_dir
index_id = args.index_id
anomaly_score = float(args.anomaly_score)
print("outpur_dir", output_dir)
print("index_id", index_id)
print("anomaly_score", anomaly_score)

print("Loading results ...")
df = run.input_datasets["interpret_input"].to_pandas_dataframe()
print(df)
print("Selecting anomalies ...")
df = df[df['score']<=anomaly_score]
print(df)

ls_anomalies = []

if df.shape[0] > 0:
    df_100 = df.sort_values("score", ascending=True).copy()
    print(df_100)

    print("Loading inference data ...")
    df_inf = run.input_datasets["inference_data"].to_pandas_dataframe()
    print(df_inf)

    print("Creating input to interpret ...")
    result_ids = df_100[index_id].values
    df_inf = df_inf.loc[df_inf[index_id].isin(result_ids),:].copy()
    X_explain = df_inf.set_index(index_id)
    print(X_explain)

    print("Loading model ...")
    model_name = 'Invoices_best_iforest.pkl'
    model_path = Model.get_model_path(model_name)
    model = joblib.load(model_path)
    print("model", model)

    print("Creating explanations ...")
    tab_explainer = TabularExplainer(model, X_explain)
    print(tab_explainer)

    # Get predictions
    predictions = model.predict(X_explain)

    # Get local explanations
    local_tab_explanation = tab_explainer.explain_local(X_explain)

    # Get feature names and importance for each possible label
    local_tab_features = local_tab_explanation.get_ranked_local_names()
    local_tab_importance = local_tab_explanation.get_ranked_local_values()

    ls_explanations = []
    for i in range(len(X_explain.index)):
        detail_id = X_explain.index[i]
        feat3, feat2, feat1 = tuple(local_tab_features[0][-3:])
        score3, score2, score1 = tuple(local_tab_importance[0][-3:])
        ls_explanations.append({
            index_id:detail_id,
            'Interpretation_Feature_1':feat1,
            'Interpretation_Score_1':score1,
            'Interpretation_Feature_2':feat2,
            'Interpretation_Score_2':score2,
            'Interpretation_Feature_3':feat3,
            'Interpretation_Score_3':score3,
        })

    df_explanations = pd.DataFrame(ls_explanations)
    print(df_explanations)
    df_results_exp = df_100.merge(df_explanations, on=index_id, how='left')
    print("Explanations:")
    print(df_results_exp)

    ls_anomalies.append(df_results_exp)

print("Consolidating all anomalies ...")
df_anomalies = pd.concat(ls_anomalies)
df_anomalies.sort_values(['score','Interpretation_Score_1'], ascending=True, inplace=True)
print(df_anomalies)

print("Adding additional columns ...")
df_anomalies = df_anomalies.merge(df_sp, on=index_id, how='left')
print(df_anomalies)

print("Saving anomalies ...")
timestr = time.strftime("%Y%m%d%H%M%S")
df_anomalies.to_csv(os.path.join(output_dir,'anomalies_'+timestr+'.csv'), index=False)

run.complete()