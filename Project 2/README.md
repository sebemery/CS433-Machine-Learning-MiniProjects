# Machine Learning Course CS-433 Project 2

In this project, we build a recommender system to predict good recommendations of item (e.g. movies) to users. Our dataset consist of 1’176’952
ratings given by 10’000 users for 1’000  items. Using different algorithms provided by different python librairies, we will try to build the most efficient recommender system as possible. We asses the performance using RMSE metric. 

## Prerequisites 
Python 3.7 installed with anaconda

## Suprise Library 
To install with anaconda run the following: 
	$ conda install -c conda-forge scikit-surprise
## FastAI Library 
To install with anaconda run the following: 
	$ conda install -c pytorch -c fastai fastai
	
## Spotlight Library 
The Spotlight functions we used are not compatible with python 3.7. We need to create a new environment  that works with python 3.6. To run Spotlight algorithm do the following: 
1. Open command line and go to the script folder
2. Create a new conda environment with python 3.6 by opening Anaconda command line and using : 
	$ conda create --name spot python=3.6
3. Activate the environment : 
	$ conda activate spot
4. Install spotlight : 
	$ conda install -c maciejkula -c pytorch spotlight
5. Run the spotlight algorithm: 
	$ python run.py 
	
## Generate the best submission
We got our best RMSE and AICROWD score with the spotlight algorithm. To generate our best submission follow the guide above for spotlight library and call the run.py file in the script folder. This creates a best_submssion.csv file in the Datasets folder suitable for submission on AICROWD. 

## Datasets Folder
- data_train.csv : Each line represents an interaction user-item (noted r44_c3, for interaction of user 44 and item 3) in the first column with the associated rating in the second column. 

- sample_submission.csv : contains the Ids (user-item) for which we have to make predictions (=recommendations) in the same format as in the data_train file.

- contains also the generated submissions files by the different models suitable for AICROWD.

## Script Folder
### Data_management.py 
- load_data_baseline: creates a sparse matrix needed to use baseline algorithms

- load_sample_sub: returns a numpy array with ids from the sample_submission.csv file

- new_dataset :  create a new csv from data_train.csv with 3 columns : "user", "item", "rating" containing only the numerical values

- create_csv_submission: generate a csv file for submission on AICROWD, given a numpy array containing the user ids, items ids and ratings associated, with the same format as sample_submission.csv

- calculate_pred_surprise : calculate predictions given a surprise algorithm and the sample_submission.csv file.

- Other functions that help load the data or generate predictions. 
See detailed descriptions of all functions in the file

### baseline.py
This file contains the baselines algorithm that calculate the general mean, user mean and item mean 

### surprise_package.py
This file contains a function (run_surprise) that generate the predicted ratings and perform cross-validation for a given surprise algorithm and another function to load the data in a surprise format. 
 
### fastai_package.py
This file contains a function (run_fastai) that generate the predicted ratings for a given fastai algorithm and another function to load the data in a surprise format. 

### data_spotlight.py
Contains all the functions that are needed to load and generate ratings with the spotlight algorithm. 

### CVspot.py 
Calculate the 5-fold cross-validaiton for the spotligh algorithm. 

### run_surprise.py 
Can be called in the command line to generate predictions with a given suprise algorithm. It creates a submission file in the Datasets folder  

### run_fastai.py 
Can be called in the command line to generate predictions with a given fastai algorithm. It creates a submission file in the Datasets folder  

### run.py
Generate predictions with spotlight algorithm and create a submission file (best_submission.csv) in the Datasets folder. This file gives us the best submission on the AICROWD competition. 

Each functions are described with more details in each file. 

Note : When running a fastai algorithm, it automatically creates a models folder in the script where some model parameters are saved. This folder has no use it can be removed when predictions are generated. 