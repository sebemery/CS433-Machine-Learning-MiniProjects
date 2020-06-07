from fastai_package import run_fastai 

"""
return an np array for a given fastai algorithm 
for EmbeddingDotBias (MF based algorithm) enter algo=1
for EmbeddingNN (NN Based algorithm) enter algo=2
"""
predictions = run_fastai('../Datasets/data_train.csv',
                          '../Datasets/rewrite_dataset.csv',
                          '../Datasets/sample_submission.csv',
                          algo=1)


"""creates a csv submission file given the ids and predictions"""
from data_management import create_csv_submission

path_submission = "../Datasets/fastai_submission.csv"

create_csv_submission(predictions, path_submission)
