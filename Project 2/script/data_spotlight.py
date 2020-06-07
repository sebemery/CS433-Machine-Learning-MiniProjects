import csv
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')
import os
import torch
import numpy as np

from data_management import new_dataset

from spotlight.datasets import _transport
from spotlight.interactions import Interactions

def load_data_spotlight(path_data,path_spotlight_data):
    """This function creates a new csv file for spotlight and load the data"""
    new_dataset(path_data,path_spotlight_data)
    dataset = get_my_own_dataset("rewrite_dataset")
    dataset.ratings = dataset.ratings.astype(np.float32)  #cast to float32 in order to use Spotlight

    return dataset

def _get_my_own(dataset):
    """ get ids and values from dataset"""
    extension = '.csv'
    URL_PREFIX = '../Datasets/'

    data = np.genfromtxt(URL_PREFIX + dataset + extension, delimiter=',', names=True, dtype=(int, int, float))
    
    users = data['user']
    items = data['item']
    ratings = data['rating']
    return (users, items, ratings)

def get_my_own_dataset(myfile):
    """
    Returns
    -------

    Interactions: :class:`spotlight.interactions.Interactions`
        instance of the interactions class
    """

    url = myfile

    return Interactions(*_get_my_own(url))