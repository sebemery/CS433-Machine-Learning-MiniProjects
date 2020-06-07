import csv
import numpy as np
import scipy.sparse as sp

def load_data_baseline(path_dataset):
    """Load data in text format, one rating per line."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)

def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()

def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        """get stats about the data"""
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_col, max_row))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings

def load_sample_sub(path_dataset):
    """Load data in text format, one rating per line, as in the competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_surprise(data)

def preprocess_surprise(data):
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), int(rating)
    data = [deal_line(line) for line in data]
    return np.array(data)
            
def new_dataset(path_data, path_rewritedata):
    """ rewrite the dataset suitable for surprise and contains only numeircal values"""
    data = load_sample_sub(path_data)
    with open(path_rewritedata, 'w', newline='') as csvfile:
        fieldnames = ['user', 'item', 'rating']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for line in data:
            writer.writerow({'user':line[0],'item':line[1],'rating':float(line[2])})

def create_csv_submission(data, name):
    """
    Creates an output file in csv format for submission to AICROWD
    Arguments: data (matrix where each row is a line of sample submission
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for line in data:
            line_id = "r{}_c{}".format(line[0], line[1])
            writer.writerow({'Id':line_id,'Prediction':int(round(line[2]))})

def calculate_pred_surprise(algo,path_sample_submission) :
    """
    Return a vector of prediction for the specified (user,item) combination
    The vector has the following structure : 
    new_sub[0] : user index, new_sub[1] : item index, new_sub[2] : prediction
    Parameters :
    algo : algorithm used to estimate the rating in the surprise library
    sample_submission : csv path of the unknown (user,item) indices to predict
    """
    sample_sub = load_sample_sub(path_sample_submission)
    new_sub = np.copy(sample_sub)
    
    for x in new_sub:
        user_index = x[0]
        item_index = x[1]
        pred = algo.predict(user_index, item_index)
        x[2] = int(round(pred[3]))
    return new_sub  