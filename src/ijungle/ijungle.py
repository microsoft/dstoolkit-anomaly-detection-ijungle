from functools import reduce
from io import BytesIO
from pyspark.sql import DataFrame
from sklearn import metrics
from sklearn.ensemble import IsolationForest

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import pyspark.sql.functions as F
import time

class iJungle():
    '''
    Class to create iJungle models. Designed to run in Spark Synapse environment.

    Example of how to run the code:
    prepped_data_path = "ijunglepaperdb.train_dataset"
    test_tablename = "ijunglepaperdb.test_dataset"

    model = iJungle(prepped_data_path, num_groups=10, train_size = 0.01, overhead_size = 0.01, seed=42)
    model.fit()
    model.predict(test_tablename)
    auc_dict = model.predict_all(test_tablename, test_frac=0.05)
    '''
    
    def __init__(self, prepped_data_path, train_size=0.0001, overhead_size=0.0001, num_groups=1, subsample_list=[4096, 2048, 1024, 512, 256], trees_list=[500, 100, 20, 10], seed=0, run_id = None):
        self.prepped_data_path = prepped_data_path
        self.train_size = train_size
        self.overhead_size = overhead_size
        self.num_groups = num_groups
        self.subsample_list = subsample_list
        self.trees_list =  trees_list
        self.seed = seed
        self.best_trees = -1
        self.best_subsample_size = -1
        self.best_group = -1
        if run_id == None:
            self.run_id = "".join(str(time.time()).split("."))
        else: self.run_id = run_id
        self.model_path = 'ijunglepaperdb.trained_models_{}'.format(self.run_id)
        self.overhead_data_path = 'ijunglepaperdb.overhead_data_path_{}'.format(self.run_id)
        self.overhead_results_path = 'ijunglepaperdb.overhead_results_path_{}'.format(self.run_id)
        self.best_model_path = 'ijunglepaperdb.best_model_path_{}'.format(self.run_id)
        self.print_params()

    def print_params(self):
        print("Training data path:", self.prepped_data_path)
        print("Run ID:", self.run_id)
        print("Model path:", self.model_path)
        print("Overhead data path:", self.overhead_data_path)
        print("Overhead results path:", self.overhead_results_path)
        print("Best model path:", self.best_model_path)

        print("Train size:", self.train_size)
        print("Overhead size:", self.overhead_size)
        print("Number of groups:", self.num_groups)
        print("Subsample list:", self.subsample_list)
        print("Trees list:", self.trees_list)
        print("Seed:", self.seed)

    def get_best(self):
        return {
            "best_trees": self.best_trees,
            "best_subsample_size":self.best_subsample_size,
            "best_group":self.best_group
        }

    def fit(self):
        self.train_models()
        print("Finished trainining models.")
        self.calculate_overhead_predictions()
        print("Finished calculating overhead predictions.")
        self.calculate_best_model()
        print("Finished calculating best model.")

    def train_models(self):
        df = spark.read.table(self.prepped_data_path)
        m = df.count()
        feats = df.columns
        print("Number of training records: {:,}".format(m))
        print("Number of features: {:,}".format(len(feats)))

        def ijungle_train(feats, seed):
            def _fun(key, pdf):
                trees = key[0]
                subsample_size = key[1]
                group = key[2]
                feats = list(pdf.columns)
                feats.remove('_group')
                feats.remove('_tree_size')
                feats.remove('_subsample_size')
                pdf = pdf[feats]
                clf = IsolationForest(
                    n_estimators = trees, 
                    max_samples=min(subsample_size, pdf.shape[0]), 
                    random_state=seed, n_jobs=-1)
                clf.fit(pdf)
                bytes_container = BytesIO()
                joblib.dump(clf, bytes_container)
                bytes_container.seek(0)
                model_bytes = bytes_container.read()
                return(pd.DataFrame([(trees, subsample_size, group, model_bytes)]))
            return(_fun)

        idx = 0
        df_list = []
        for trees in self.trees_list:
            for subsample_size in self.subsample_list:
                for group in range(self.num_groups):
                    df_group = df.sample(withReplacement=False, fraction=self.train_size, seed=self.seed+idx)
                    df_group = df_group.withColumn('_tree_size', F.lit(trees))
                    df_group = df_group.withColumn('_subsample_size', F.lit(subsample_size))
                    df_group = df_group.withColumn('_group', F.lit(group))
                    df_list.append(df_group)
                    idx += 1

        #https://stackoverflow.com/questions/54229806/efficiently-merging-a-large-number-of-pyspark-dataframes
        def pairwise_reduce(op, x):
            while len(x) > 1:
                v = [op(i, j) for i, j in zip(x[::2], x[1::2])]
                if len(x) > 1 and len(x) % 2 == 1:
                    v[-1] = op(v[-1], x[-1])
                x = v
            return x[0]
        df_all = pairwise_reduce(DataFrame.unionAll, df_list)

        tot_models = idx 
        print(f'Number of models: {tot_models}')
        print("Number of records: {:,}".format(df_all.count()))

        df_models = df_all.groupBy('_tree_size', '_subsample_size', '_group').applyInPandas(
            ijungle_train(feats, self.seed), 
            schema="tree_size long, subsample_size long, id long, model binary"
        )
        df_models.write.mode('overwrite').saveAsTable(self.model_path)

    def calculate_overhead_predictions(self):
        df = spark.read.table(self.prepped_data_path)
        
        df_models = spark.read.table(self.model_path)
        df_models = df_models.withColumn("model_id",F.monotonically_increasing_id())
        ids = df_models.select('model_id').collect()
        ids = sorted(ids)
        num_models = len(ids)

        df_W = df.sample(withReplacement=False, fraction=self.overhead_size, seed=self.seed)
        df_W = df_W.withColumn("overhead_id", F.monotonically_increasing_id())
        print("Number of records of overhead dataset: {:,}".format(df_W.count()))
        df_W.write.mode('overwrite').saveAsTable(self.overhead_data_path)

        #This function takes a dictionary of models and runs them across a pandas dataframe from mapInPandas
        def clf_predict():
            def _fun(iterator):
                for pdf in iterator:
                    pdf_out_exists = False
                    for key in clf_dict:
                        clf = clf_dict[key]
                        pdf.set_index(["overhead_id"], inplace=True)
                        _predict = clf.score_samples(pdf)
                        pdf.reset_index(drop=False, inplace=True)
                        pdf_temp = pdf[["overhead_id"]]
                        pdf_temp['tree_size'] = [key[0]]*pdf.shape[0]
                        pdf_temp['subsample_size'] = [key[1]]*pdf.shape[0]
                        pdf_temp['group_num'] = [key[2]]*pdf.shape[0]
                        pdf_temp["predict"] = _predict
                        if pdf_out_exists==False:
                            pdf_out = pdf_temp
                            pdf_out_exists = True
                        else:
                            pdf_out = pd.concat([pdf_out,pdf_temp])
                    yield(pdf_out)
            return(_fun)

        #The main idea of this piece of code is: Loading models has overhead. So load many at once. Running models in mapInPandas has overhead, so run many at once.
        #The while loop does both of these things. It loads a 'chunk' of models with one collect, creates a python dictionary with the models and the parameters to the models
        #and runs those models using mapInPandas and the clf_predict function. It also writes out the models for each iteration of the while loop, so that the process DAG
        #doesn't get too complicated if the write happens after each call of mapInPandas.
        #This M factor is very important. M is the number of models loaded into memory at once. Too many and the executors will run out of memory. But too few and the job won't run optimally.
        M = 250
        chunk = 0
        schema = "overhead_id integer, tree_size integer, subsample_size integer, group_num integer, predict float"
        df_predict_exists = False
        while M*chunk < num_models:
            rows = ids[M*chunk:(chunk+1)*M]
            chunk_models = df_models.where(F.col("model_id") >= rows[0][0]).where(F.col("model_id") <= rows[-1][0]).collect()
            clf_dict = {}
            for model in chunk_models:
                clf_dict[(model.tree_size, model.subsample_size, model.id)] = joblib.load(BytesIO(model.model))
            df_predict = df_W.mapInPandas(clf_predict(), schema = schema)
            if not df_predict_exists:
                df_predict_exists = True
                df_predict.write.mode('overwrite').saveAsTable(self.overhead_results_path)
            else:
                try:
                    df_predict.write.mode('append').saveAsTable(self.overhead_results_path)
                except:
                    print("Error in writing df_predict in calculate_overhead_predictions.")
                    print("Number of models in chunk: {:,}".format(len(clf_dict.keys())))
                    print("Models in chunk: {}".format(clf_dict.keys()))
            chunk += 1

    def calculate_best_model(self):
        df_predict = spark.read.table(self.overhead_results_path)
        df_avg = df_predict.groupBy('overhead_id').agg((F.sum('predict')/F.count('predict')).alias('avg'))
        df_predict = df_predict.join(df_avg,on='overhead_id')
        df_predict = df_predict.withColumn("squared_residuals",F.pow(F.col('predict') - F.col("avg"),2))
        df_model = df_predict.groupBy("tree_size","subsample_size","group_num").agg(F.sum("squared_residuals").alias("sum_of_squared_residuals"))
        df_model = df_model.orderBy("sum_of_squared_residuals",ascending=True)
        best_trees, best_subsample_size, best_group, _ = df_model.take(1)[0]
        print("Best iForest: {}, {}, {}".format(best_trees, best_subsample_size, best_group))
        df_iFor = spark.read.table(self.model_path)
        self.best_group = best_group
        self.best_trees = best_trees
        self.best_subsample_size = best_subsample_size
        model_bytes = df_iFor.where((F.col('id')==best_group)&(F.col('tree_size')==best_trees)&(F.col('subsample_size')==best_subsample_size)).select('model').collect()[0]['model']
        spark.createDataFrame([('best_iforest',model_bytes)],schema=['id','model']).write.mode('overwrite').saveAsTable(self.best_model_path)

    def predict(self, test_data_path, id_feat=['isAnomaly'], id_feat_types=['long']):
        model_bytes = spark.read.table(self.best_model_path).take(1)[0]['model']
        clf = joblib.load(BytesIO(model_bytes))
        df_predict = spark.read.table(test_data_path)
        
        def ijungle_predict(id_feat, clf):
            def _fun(iterator):
                for pdf in iterator:
                    pdf.set_index(id_feat, inplace=True)
                    _predict = clf.predict(pdf)
                    _score = clf.score_samples(pdf)
                    pdf.reset_index(drop=False, inplace=True)
                    pdf_out = pd.DataFrame()
                    pdf_out[id_feat] = pdf[id_feat]
                    pdf_out['predict'] = _predict
                    pdf_out['score'] = _score
                    yield(pdf_out)
            return(_fun)

        dcc_str = ", ".join([x[0]+" "+x[1] for x in zip(id_feat, id_feat_types)]) + ", predict int, score float"
        df_results = df_predict.mapInPandas(ijungle_predict(id_feat, clf),dcc_str)
        predictions_path = test_data_path + '_predictions_{}'.format(self.run_id)
        print("Path for model predictions:", predictions_path)
        df_results.write.mode('overwrite').saveAsTable(predictions_path)

    def predict_all(self, test_data_path, gt_feat='isAnomaly', test_frac = 1, save_hist=True):
        df = spark.read.table(test_data_path)
        df = df.sample(withReplacement=False, fraction=test_frac, seed=self.seed)
        print("The number of rows in the sampled test dataset area:", df.count())

        df_models = spark.read.table(self.model_path)
        df_models = df_models.withColumn("model_id",F.monotonically_increasing_id())
        ids = df_models.select('model_id').collect()
        ids = sorted(ids)
        num_models = len(ids)

        #This function takes a dictionary of models and runs them across a pandas dataframe from mapInPandas
        def clf_predict():
            def _fun(iterator):
                for pdf in iterator:
                    pdf_out_exists = False
                    for key in clf_dict:
                        clf = clf_dict[key]
                        y_test = (pdf[gt_feat] * (-2)) + 1
                        pdf.set_index([gt_feat], inplace=True)
                        _predict = clf.score_samples(pdf)
                        pdf.reset_index(drop=False, inplace=True)
                        pdf_temp = pd.DataFrame()
                        pdf_temp['tree_size'] = [key[0]]*pdf.shape[0]
                        pdf_temp['subsample_size'] = [key[1]]*pdf.shape[0]
                        pdf_temp['group_num'] = [key[2]]*pdf.shape[0]
                        pdf_temp['predict'] = _predict
                        pdf_temp['ground_truth'] = y_test
                        if pdf_out_exists==False:
                            pdf_out = pdf_temp
                            pdf_out_exists = True
                        else:
                            pdf_out = pd.concat([pdf_out,pdf_temp])
                    yield(pdf_out)
            return(_fun)

        predict_all_path = test_data_path + '_all_predictions_{}'.format(self.run_id)

        #This M factor is very important. M is the number of models loaded into memory at once. Too many and the executors will run out of memory. But too few and the job won't run optimally.
        M = 250
        chunk = 0
        schema = "tree_size integer, subsample_size integer, group_num integer, predict float, ground_truth float"
        df_predict_exists = False
        while M*chunk < num_models:
            rows = ids[M*chunk:(chunk+1)*M]
            chunk_models = df_models.where(F.col("model_id") >= rows[0][0]).where(F.col("model_id") <= rows[-1][0]).collect()
            clf_dict = {}
            for model in chunk_models:
                clf_dict[(model.tree_size, model.subsample_size, model.id)] = joblib.load(BytesIO(model.model))
            df_predict = df.mapInPandas(clf_predict(), schema = schema)
            if not df_predict_exists:
                df_predict_exists = True
                df_predict.write.mode('overwrite').saveAsTable(predict_all_path)
            else:
                try:
                    df_predict.write.mode('append').saveAsTable(predict_all_path)
                except:
                    print("Error in writing df_predict in calculate_overhead_predictions.")
                    print("Number of models in chunk: {:,}".format(len(clf_dict.keys())))
                    print("Models in chunk: {}".format(clf_dict.keys()))
            chunk += 1

        pdf_test = spark.read.table(predict_all_path)

        def grid_auc():
            def _fun(key, pdf):
                trees = key[0]
                subsample_size = key[1]
                group = key[2]
                y_scores = pdf['predict']
                y_test = pdf['ground_truth']
                auc = metrics.roc_auc_score(y_test, y_scores)
                return(pd.DataFrame([{
                    'tree_size':trees, 
                    'subsample_size':subsample_size,
                    'group':group,
                    'auc':auc}]))
            return(_fun)

        df_auc = pdf_test.groupBy('tree_size', 'subsample_size', 'group_num').applyInPandas(
            grid_auc(), 
            schema="tree_size integer, subsample_size integer, group integer, auc double"
        )

        auc_path = test_data_path + '_auc_{}'.format(self.run_id)
        df_auc.write.mode('overwrite').saveAsTable(auc_path)
        best_model_params = self.get_best()
        trees, subsample_size, group = best_model_params['best_trees'], best_model_params['best_subsample_size'], best_model_params['best_group']
        df_auc = spark.read.table(auc_path).toPandas()
        best_auc = df_auc[(df_auc.tree_size==trees)&(df_auc.subsample_size==subsample_size)&(df_auc.group==group)].auc.tolist()[0]
        median_auc = df_auc.auc.median()
        mean_auc = df_auc.auc.mean()
        
        if save_hist:
            plt.figure(figsize=(10,10),facecolor='w')
            plt.hist(df_auc.auc,bins=200,alpha=0.5,label="All models")
            plt.title("Distribution of AUC score for {} models in run {}".format(df_auc.shape[0], self.run_id),fontsize=15)
            best_model_params = self.get_best()
            trees, subsample_size, group = best_model_params['best_trees'], best_model_params['best_subsample_size'], best_model_params['best_group']
            best_auc = df_auc[(df_auc.tree_size==trees)&(df_auc.subsample_size==subsample_size)&(df_auc.group==group)].auc
            plt.vlines(best_auc,0,25,'r',linewidth=3,alpha=0.5, label="Best model ({}, {}, {})".format(trees, subsample_size, group))
            plt.vlines(df_auc.auc.mean(),0,25,'purple',linewidth=3,alpha=0.5, label="Average model".format(trees, subsample_size, group))
            plt.vlines(df_auc.auc.median(),0,25,'green',linewidth=3,alpha=0.5, label="Median model".format(trees, subsample_size, group))
            plt.legend(loc=2,fontsize=15)
            plt.xlabel("AUC score",fontsize=15)
            plt.ylabel("Number of models",fontsize=15)
            plt.show()

        return {'best_auc':best_auc, 'median_auc':median_auc, 'mean_auc':mean_auc}
