import numpy as np
import scipy.sparse as sp



def baseline_global_mean(train, test):
    """baseline method: use the global mean."""
    
    # find the non zero ratings in the train
    nonzero_train = train[train.nonzero()]

    # calculate the global mean
    global_mean_train = nonzero_train.mean()

    # find the non zero ratings in the test
    #todense return a dense numpy array matrix of the sparse matrix
    nonzero_test = test[test.nonzero()].todense()

    # predict the ratings as global mean
    mse = calculate_mse(nonzero_test, global_mean_train)
    rmse = np.sqrt(1.0 * mse / nonzero_test.shape[1])
    print("test RMSE of baseline using the global mean: {v}.".format(v=rmse))

def baseline_user_mean(train, test):
    """baseline method: use the user means as the prediction."""
    
    mse = 0
    num_items, num_users = train.shape
    
    for user_idx in range(num_users) :
        user_train = train[:,user_idx]
        # find the non zero ratings in the train
        nonzero_train_user = user_train[user_train.nonzero()]
        
        if nonzero_train_user.get_shape()[1] !=0 :
            # calculate the global mean
            user_train_mean = nonzero_train_user.mean()
        else:
            continue
        
        # find the non zero ratings in the test
        user_test = test[:,user_idx]
        nonzero_test_user = user_test[user_test.nonzero()].todense()
        
        # predict the ratings as user mean
        mse += calculate_mse(nonzero_test_user, user_train_mean)
    
    rmse = np.sqrt(1.0 * mse / test.nnz)
    print("test RMSE of baseline using the global mean: {v}.".format(v=rmse))

def baseline_item_mean(train, test):
    """baseline method: use item means as the prediction."""
    mse = 0
    num_items, num_users = train.shape
    
    for items_idx in range(num_items) :
        item_train = train[items_idx,:]
        # find the non zero ratings in the train
        nonzero_train_item = item_train[item_train.nonzero()]
        
        if nonzero_train_item.get_shape()[1] !=0 :
            # calculate the global mean
            item_train_mean = nonzero_train_item.mean()
        else :
            continue
        
        # find the non zero ratings in the test
        item_test = test[items_idx,:]
        nonzero_test_item = item_test[item_test.nonzero()].todense()
        
        # predict the ratings as user mean
        mse += calculate_mse(nonzero_test_item, item_train_mean)
    
    rmse = np.sqrt(1.0 * mse / test.nnz)
    print("test RMSE of baseline using the global mean: {v}.".format(v=rmse))

def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)