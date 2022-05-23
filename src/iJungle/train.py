# -*- coding: utf-8 -*-
from iJungle.config import _MODEL_DIR

import random
from sklearn.ensemble import IsolationForest
import joblib
import os
import numpy as np
import pandas as pd
from numpy import linalg as LA

def model_train_fun(df, trees=100, subsample_size=8192, train_size = 0.2, max_sss=8192, verbose=True):
    """Function to train the model

    As the model implies, this function will be used to train the model.
    It takes a dataFrame input and two optional parameters as the 
    iForest hyper parameters (e.g. trees and subsample_size)

    The number of registers in the df dataFrame are retrieved (df_len)
    and a list from 0 to this number is created. The list is then 
    shuffled, which will be used for sampling df in the following.

    Some variables are initialized and a loop starts traversing the
    data set. At each iteration, using the randomly shuffled list 
    (my_indexes), the dataset is sampled by a number of samples equal to
    subsample_size. Then, an iForest model is created using the provided
    hyperparameters and the corresponding subsample. This model is
    stored into a dictionary (iFor_dic) that is what is finally returned
    by the function.

    Keyword arguments:
    :param df: input pandas dataframe
    :param trees: number of trees to train each iForest
    :param subsample_size: max_samples parameter to train each iForest
    :param n: percentage of data to use for training
    :param max_sss: maximum sub-sample size
    :return: list of trained iForrests
    """
    # TODO: Update documentation
    try:
        assert(train_size > 0.0 and train_size <= 1.0)
        os.makedirs(_MODEL_DIR, exist_ok=True)

        df_len = int(np.floor(len(df.index)*train_size))
        my_indexes = list(range(0, df_len))
        random.shuffle(my_indexes)

        rng = np.random.RandomState(42)

        iFor_list = []
        i, counter = 0, 0

        while i < df_len:
            sub_sample = df.iloc[my_indexes[i:i+subsample_size]]
        
            clf = IsolationForest(
                n_estimators = trees, 
                max_samples=min(subsample_size,len(sub_sample)), 
                random_state=rng, n_jobs=-1)
            clf.fit(sub_sample)

            iFor_list.append(clf)
            counter += 1
            i += max_sss
            # TODO: implement logger
            if verbose:
                print("{}/{}".format(counter, int(df_len/max_sss+1)))
            
        filename = 'iJungle_light_' + str(trees) + '_' + str(subsample_size) + '.pkl'
        joblib.dump(value=iFor_list, filename=os.path.join(_MODEL_DIR, filename))
 
        return(filename)
    except Exception as err:
        # TODO: Implement logger
        if verbose:
            print(err)

    return(1)

def grid_train(df, subsample_list = [4096, 2048, 1024, 512],
               trees_list = [500, 100, 20, 10], train_size=0.2,
               verbose=True):
    """Grid Isolation Jungle train
    
    Train Isolation jungle with different subsamples and trees
    hyper-parameters

    Keyword arguments:
    :param subsample_list: different subsamples to test
    :param trees_list: number of estimators to try
    :param n_jobs: number of max parallel jobs
    """
    if not os.path.isdir(_MODEL_DIR):
        os.makedirs(_MODEL_DIR)

    model_filenames = []
    for subsample_size in subsample_list:
        for trees in trees_list:
            # TODO: Implement logger
            if verbose:
                print('Iteration: Subsample = {}, trees= {}'.format(subsample_size, trees))
            model_filenames.append(model_train_fun(df, trees, subsample_size, train_size, max(subsample_list), verbose))
    
    return(model_filenames)


def model_eval_fun(df, iFor_list, verbose=True):
    # TODO: This can be explored later: 
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
    results = np.zeros((len(df), len(iFor_list)))
    for i, clf in enumerate(iFor_list):
        results[:, i] = clf.predict(df)
        # TODO: implement logger
        if verbose:
            print("{}/{}".format(i+1, len(iFor_list)))
    return(results)


def grid_eval(df, subsample_list = [4096, 2048, 1024, 512],
               trees_list = [500, 100, 20, 10], overhead_size=0.2,
               verbose=True):
    # Origial code calculated from the rest of the records not used for training a top of overhead_size
    # samples to be used to calculate the most representative iForrest.  As each model_train shuffles the
    # dataset again, it is not different than selecting a overhead_size sample.  Even better, overhead_size
    # should be a percentage of the total number of samples
    try:
        assert(overhead_size > 0.0 and overhead_size <= 1.0)
        os.makedirs(_MODEL_DIR, exist_ok=True)
        
        df_len = int(np.floor(len(df.index)*overhead_size))
        my_indexes = list(range(0, df_len))
        random.shuffle(my_indexes)
        
        W = df.iloc[my_indexes[:df_len]]
        
        results_dic = {}
        ## Evaluation with stored models as external files(joblib format)
        for i, subsample_size in enumerate(subsample_list):
            results_dic_t = {}
            for j, trees in enumerate(trees_list):
                filename = 'iJungle_light_' + str(trees) + '_' + str(subsample_size) + '.pkl'
                # TODO: Implement logger
                if verbose:
                    print('Reading ' + filename)
                iFor_list = joblib.load(os.path.join(_MODEL_DIR, filename))
                results_dic_t[str(trees)] = model_eval_fun(W, iFor_list, verbose)
            results_dic[str(subsample_size)] = results_dic_t
        
        filename_results = 'iJungle_light_results_overhead.pkl'
        
        results = pd.DataFrame(results_dic)
        joblib.dump(value=results, filename=os.path.join(_MODEL_DIR, filename_results))
        return(results)
    except Exception as err:
        # TODO: Implement logger
        print(err)
    return(1)


def get_grid_eval_results(verbose = True):
    picklename = os.path.join(_MODEL_DIR, 'iJungle_light_results_overhead.pkl')
    if os.path.exists(picklename):
        if verbose:
            print("Reading ", picklename)
        results = joblib.load(picklename)
        return(results)
    else:
        raise Exception("grid_eval has not have been executed")

def best_iforest_params(results, verbose=True):
    results_av = results.copy()
    if verbose:
        print("Shape of results:",results.shape,"with each element of size", results.iloc[0,0].shape)
    for ss in results.columns:
        for tr in results.index:
            results_av[ss][tr] = np.average(results[ss][tr], 1)
    if verbose:
        print("Shape of results_av:",results_av.shape,"with each element of size", results_av[ss][tr].shape)
    av = np.average(np.average(results_av, 0))
    if verbose:
        print("Shape of av",av.shape)
        print('Number of anomalies with score = -1: {}'.format(len(av[av == -1])))
    ## Select the best model under L2 metric with grid-search
    best_l2 = np.inf
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            for k in range(results.iloc[0,0].shape[1]):
                if LA.norm(results.iloc[i, j][:, k]-av) <= best_l2:
                    ## Memorize threshold
                    best_l2 = LA.norm(results.iloc[i, j][:, k]-av)
                    ## Memorize position corresponding to best params
                    best_iF_i, best_iF_j, best_iF_k = i, j, k

    if verbose:
        print("Best subsample:", results.columns[best_iF_j])
        print("Best number of trees:", results.index[best_iF_i])
        print("Best iForest:", best_iF_k)
    
    subsample_size = results.columns[best_iF_j]
    trees = results.index[best_iF_i]
    return(subsample_size, trees, best_iF_k)

def best_iforest(results, verbose=True):
    ## Select the best parameter of generated models
    subsample_size, trees, best_iF_k = best_iforest_params(results, verbose)

    picklename = os.path.join(_MODEL_DIR,'iJungle_light_' + str(trees) + '_' + str(subsample_size) + '.pkl')
    if verbose:
        print('Reading ' + picklename)
    iFor_list = joblib.load(picklename)    

    model = iFor_list[best_iF_k]
    if verbose:
        print("Model selected!")
    return(model)


def train_bundle(df, subsample_list = [4096, 2048, 1024, 512],
               trees_list = [500, 100, 20, 10], train_size=0.2, overhead_size=0.2,
               verbose=True):
    ## Generate & save models as external files(pickle format) in accordance with subsample_list, trees_list, train_size
    grid_train(df, subsample_list, trees_list, train_size, verbose)
    ## Evaluation with trained models as external files
    results = grid_eval(df, subsample_list, trees_list, overhead_size, verbose)
    ## Select the best model
    model = best_iforest(results, verbose)
    return(model)

def select_overhead_data(df, overhead_size=0.2, verbose=True):
    df_len = int(np.floor(len(df.index)*overhead_size))
    my_indexes = list(range(0, df_len))
    random.shuffle(my_indexes)
    W = df.iloc[my_indexes[:df_len]].copy()
    if verbose:
        print("Overhead data shape:", W.shape)
    return(W)

