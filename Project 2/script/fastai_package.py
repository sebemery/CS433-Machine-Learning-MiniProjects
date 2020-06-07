import numpy as np
import pandas as pd
import fastai
from fastai.collab import *
from data_management import*

def load_data_fastai(path_data, path_rewritedata):
    """Create a new csv for fastai and load the data"""
    new_dataset(path_data, path_rewritedata)
    data = pd.read_csv(path_rewritedata)
    data2 = CollabDataBunch.from_df(data, seed=12, valid_pct=0.15, user_name='user', item_name='item', rating_name='rating')
    data2.show_batch()
    return data2

def run_fastai(path_data,rewrite_dataset,path_sample_submission, algo):
    """ This function runs fastai algorithms.
    Given algo=1 : runs the fastai embeddingDotBias algorithm, which is a MF based algorithm
    Given algo=2 : runs the embeddingNN algorithm, which is a NN bassed algorithm

    The function return an numpy array with ids and predictions for sample_submission ids
    """
    data = load_data_fastai(path_data,rewrite_dataset)
    
    #EmbeddingDotBias algorithm   
    if algo == 1:
        learn = collab_learner(data, n_factors=200, y_range=[1,5], wd=5e-2)
        learn.lr_find() # find learning rate
        learn.recorder.plot() # plot learning rate graph
        learn.fit_one_cycle(10, 3e-4)
    
    #EmbdedingNN algorithm
    elif algo == 2:
        learn = collab_learner(data, use_nn=True, 
                                     emb_szs={'user': 100, 'item':100}, 
                                     layers=[32, 16], 
                                     y_range=[1,5])
        learn.lr_find() # find learning rate
        learn.recorder.plot() # plot learning rate graph
        learn.fit_one_cycle(5, 5e-2)
    else:
        print('algo only takes value 1 for embeddingsDotBias algorithm and 2 for enbeddingNN algorithm')
        return
    
    #Load ids from sample_submission file
    sample_sub  = load_sample_sub(path_sample_submission)
    
    fastai_sub = np.copy(sample_sub)
    
    #Calculate predictions for sample_sub ids
    preds = learn.model(torch.tensor(fastai_sub[:,0]),torch.tensor(fastai_sub[:,1]))
    
    for ind, p in enumerate(list(zip(preds))):
        fastai_sub[ind,2] = round(p[0].item())
    
    #return numpy array with ids and predictions        
    return fastai_sub