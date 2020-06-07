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

from spotlight.factorization.explicit import ExplicitFactorizationModel

model = ExplicitFactorizationModel(loss='regression',
                                   embedding_dim=8,  # latent dimensionality
                                   n_iter=200,  # number of epochs of training
                                   batch_size=4096,  # minibatch size
                                   l2=5e-5,  # strength of L2 regularization
                                   learning_rate=2e-3,  #learning rate
                                   use_cuda=torch.cuda.is_available(),  #uses GPU 
                                   random_state=np.random.RandomState(42)  #fixed seed for reproducibility
                                  )

model.fit(dataset, verbose=True)  #verbose=False if you don't want to see any message during the process

#Load ids for submission
path_sample_sub = "../Datasets/sample_submission.csv"
sample_sub = load_sample_sub(path_sample_sub)

#calculate predictions
predictions = model.predict(sample_sub[:,0], sample_sub[:,1])

"""Here we create the new submission file by copying the sample one and inserting the new prediction into.
We round the values to integers and make sure they are between 1 and 5"""
spotlight_explicit_sub = np.copy(sample_sub)

for i, x in enumerate(spotlight_explicit_sub):
    pred = int(round(predictions[i]))
    if(pred < 1): 
        pred = 1
    elif(pred > 5):
        pred = 5
    x[2] = pred


""" Creating submisison """
from data_management import create_csv_submission

print("Creating csv file") 

path_submission = "../Datasets/best_submission.csv"

create_csv_submission(spotlight_explicit_sub, path_submission)

