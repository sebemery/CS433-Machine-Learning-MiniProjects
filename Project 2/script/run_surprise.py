#Lists of surprise algorithms that can be tested
algonames = ['Baseline','SlopeOne','KNNBasic','KNNBaseline','CoClustering','SVD','SVDpp']

from surprise_package import run_surprise

algoname=algonames[0]

#return ids, and predicted ratings for sample_submission 
predictions = run_surprise('../Datasets/data_train.csv',
                           '../Datasets/rewrite_dataset.csv',
                           '../Datasets/sample_submission.csv',    
                           algoname=algoname, cross_validation=True)

print("predictions shape: ")
print(predictions.shape)

print("Some predictions: ")
print(predictions[0:20])

from data_management import create_csv_submission

path_submission = "../Datasets/surprise_submission.csv"

create_csv_submission(predictions, path_submission)