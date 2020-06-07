import numpy as np
import surprise
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly
from surprise import SlopeOne
from surprise import KNNBasic
from surprise import KNNBaseline
from surprise import CoClustering
from surprise import SVD
from surprise import SVDpp
from surprise.model_selection import cross_validate
from data_management import *

def load_data_surprise(path_data, path_rewritedata):
    """
    Returns a new Dataset object compatible with Surprise functions. 
    In the process it creates a new csv file with the modified initial dataset.
    path_data: path of the existing dataset
    new_path: where to store the new dataset
    """
    new_dataset(path_data, path_rewritedata)
    # As we're loading a custom dataset, we need to define a reader.
    reader = Reader(line_format='user item rating', sep=',',skip_lines=1)
    # We load the dataset we created
    data = Dataset.load_from_file(path_rewritedata, reader=reader)
    return data

def run_surprise(path_data,rewrite_dataset,path_sample_submission, algoname, cross_validation=True):
    """ 
    This function return the ids and predictions for sample_submission ids,
    
    1st Loads the data by creating a new csv file suitable for surprise functions

    Then predict the ratings using the algortihm given by algoname

    Perform cross-validation if cross_validation parameter is True

    For each algorithm, we initilaize them with the parameters 
    that gave us the best results
    """

    if algoname == 'Baseline':
        algo = BaselineOnly()
        
    elif algoname == 'SlopeOne':
        algo = SlopeOne()
        
    elif algoname == 'KNNBasic':
        sim_options = {'name': 'msd',
               "min_support": 1,
               'user_based': False  # compute  similarities between items
               }
        algo = KNNBasic(k=40, min_k=3, sim_options=sim_options)
        
    elif algoname == 'KNNBaseline':
        sim_options = {'name': 'msd',
               "min_support": 1,
               'user_based': False  # compute  similarities between items
               }
        algo = KNNBaseline(k=40, min_k=3, sim_options=sim_options)
        
    elif algoname == 'CoClustering':
        algo = CoClustering()
        algo.n_cltr_u = 2
        algo.n_cltr_i = 500
        
    elif algoname == 'SVD':
        algo = SVD()
        algo.lr_bu = 0.007
        algo.lr_bi = 0.0008
        algo.lr_pu = 0.01
        algo.lr_qi = 0.02
        algo.reg_bu = 0.1
        algo.reg_bi = 0.0001
        algo.reg_pu = 0.8
        algo.reg_qi = 0.01
        
    elif algoname == 'SVDpp':
        algo = SVDpp()
        algo.lr_bu=0.005
        algo.lr_bi=0.0001
        algo.lr_pu=0.01
        algo.lr_qi=0.01
        algo.reg_bu=0.01
        algo.reg_bi=0.01
        algo.reg_pu=0.5
        algo.reg_qi=0.01
        algo.n_factors=200
    
    #return an error if algoname is not on the list above   
    else : 
        print('ERROR: Invalid algoname')
        return  
    
    #Load data
    data = load_data_surprise(path_data, rewrite_dataset)
    trainingSet = data.build_full_trainset()
    algo.fit(trainingSet)
    
    #Cross-validates if cross_validation parameter is true
    if cross_validation:
        cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)
    
    #return np array with ids and predicted ratings for sample_submission
    return calculate_pred_surprise(algo, path_sample_submission)