import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')

from data_management import load_sample_sub  
from data_spotlight import *

"""------------------------------------------LOOK HERE---------------------------------------------------------------
Procedure to make it work:
create a new conda environment with python = 3.6 by opening Anaconda command line and using: "conda create --name spot python=3.6"
activate the environment: "conda activate spot"
install spotlight: "conda install -c maciejkula -c pytorch spotlight"
go to the folder where the file is and use "python run.py" 

This is necessary because Spotlight package is not compatible with Python 3.7

"""
path_data = "../Datasets/data_train.csv"
path_spotlight_data = "../Datasets/rewrite_dataset.csv"

#This creates the new csv file used by Spotlight and load the data into dataset.
dataset = load_data_spotlight(path_data,path_spotlight_data)

from spotlight.cross_validation import random_train_test_split
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.evaluation import rmse_score

model = ExplicitFactorizationModel(loss='regression',
                                   embedding_dim=8,  # latent dimensionality
                                   n_iter=200,  # number of epochs of training
                                   batch_size=4096,  # minibatch size
                                   l2=5e-5,  # strength of L2 regularization
                                   learning_rate=2e-3,  #learning rate
                                   use_cuda=torch.cuda.is_available(),  #uses GPU 
                                   random_state=np.random.RandomState(42)  #fixed seed for reproducibility
                                  )

rmse_scores = []

"""
since the spotlight library only have a function to split the data 
in a random manner we calculate the cv by split 5 times the data 
in a random manner and we take the average.
"""
for i in range(5): 
	train, test = random_train_test_split(dataset, test_percentage=0.2)
	model.fit(train, verbose=True)  #verbose=False if you don't want to see any message during the process
	rmse = rmse_score(model,test)
	rmse_scores.append(rmse)

print(rmse_scores)

meanRMSE = sum(rmse_scores)/5.0

print(meanRMSE)