import numpy as np

"""
data processing related methods
* train/test split
* cross-validation data split
"""

def build_label(data):
    """extract label from original data, second column is label by default
    
    :param data: original data from csv
    """
    label = data[:,1]
    dict_x = {'s': 1, 'b': -1} 
    for i in dict_x.keys():
        label[label==i]=dict_x.get(i)
    return label.astype('float64')

def data_preprocessing(data, nan = "delete", normalize = True, is_test=False):
    """preprocessing data
    read in example:
    data = np.genfromtxt("./train.csv", skip_header=1, delimiter = ",")
    :param data: original data from csv
    :param nan: method to deal with NAN value (delete,mean,median)
    :nomalize: whether normalize the data
    """
    #one-hot
    classes = len(np.unique(data[:,22]))
    targets = data[:,22].reshape(-1)
    targets = targets.astype("int",copy=False)
    one_hot_targets = np.eye(len(targets), classes)[targets]  
    data = np.delete(data, 22, axis = 1)

    #missing value
    data[data == -999] = np.nan
    data[data == 0.0] = np.nan
    data = data.astype(np.float64)

    if nan == "delete":
        delete = []
        for i in range(0,data.shape[1]):
            if np.any(np.isnan(data[:,i])):
                delete.append(i)   
        data = np.delete(data, delete, axis = 1)
    elif nan == "mean":
        for i in range(0,data.shape[1]):
            np.nan_to_num(data[:,i],nan=np.nanmean(data[:,i]),copy = False)
    elif nan == "median":
        for i in range(0,data.shape[1]):
            np.nan_to_num(data[:,i],nan=np.nanmedian(data[:,i]),copy = False)
    else:
        raise Exception("Method not defined")
    if normalize:
        for i in range(0, data.shape[1]):
            if i == 22 and nan != "delete":
                continue  
            mean = np.mean(data[:,i])
            std = np.std(data[:,i])
            data[:,i] = (data[:,i]-mean) / std
            # min_value = min(data[:,i])
            # max_value = max(data[:,i])
            # data[:,i] = (data[:,i]-min_value)/(max_value-min_value)

    
    if is_test:
        return data

    X = np.concatenate([data,one_hot_targets], axis = 1)
    return X

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    
    :param x: input data of shape (N, D)
    :param y: label data of shape (N,)
    :param ratio: ratio of training data
    :return x_train, x_test, y_train, y_test
    """
    # set seed
    np.random.seed(seed)
    N = x.shape[0]  # Sample num
    shuffle_index = np.random.permutation(N)
    tr_N = int(ratio * N)  # training data num
    tr_index = shuffle_index[:tr_N]  # trainining data index
    te_index = shuffle_index[tr_N:]
    
    x_train, x_test = x[tr_index], x[te_index]
    y_train, y_test = y[tr_index], y[te_index]
    return x_train, x_test, y_train, y_test

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

		:param x: input data of shape (N, D)
		:param degree: maximum degree for expansion
		:return matrix: expanded data of shape [sample, degree]
		"""
    # polynomial basis function:
    poly_x = []
    for j in range(degree+1):
        poly_x.append(x**j)

    return np.concatenate(poly_x, axis=1)  # shape [sample, degree]

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def get_cross_validation_data(y, x, k, degree, seed, k_fold=10):
    """return the cross validation data.
    
    :param k_fold: cross validation fold number
    :param k: k-th fold for validation
    :param degree: maximum degree for expansion
    """
    # get k indices
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    # get k'th subgroup in test, others in train:
    tr_indices = np.concatenate([k_indices[i] for i in range(len(k_indices)) if i != k], axis=0)
    te_indices = k_indices[k]
    x_te, y_te = x[te_indices], y[te_indices]
    x_tr, y_tr = x[tr_indices], y[tr_indices]

    # form data with polynomial degree:
    x_tr = build_poly(x_tr, degree)
    x_te = build_poly(x_te, degree)
    
    return x_tr, x_te, y_tr, y_te

"""
Model optimization related methods
* loss functions
* gradient functions
"""

def compute_gradient(y, tx, w):
    """Calculate the gradient.

    :param y: label data of shape (N,)
    :param tx: input data of shape (N, D)
    :param w: model weights of shape (1, D)
    :return: gradient
    """
    e = y - tx.dot(w).squeeze()
    grad = -tx.T.dot(e)/len(e)
    return grad

def sigmoid(x):
    """Calculate Sigmoid
    :param xt: input data of shape (N, 1)
    :param w: model weights of shape (1, D)
    :return: sigmoid
    """

    return 1.0 / (1 + np.exp(-x))

def compute_loss_lr(y, tx, w):
    """compute the cost by negative log likelihood. 
    cost = -Y'log(H) - (1 - Y')log(1 - H)

    :param y: label data of shape (N, 1)
    :param tx: input data of shape (N, D)
    :param w: model weights of shape (1, D)
    """
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def compute_gradient_lr(y, tx, w):
    """Calculate the gradient.

    :param y: label data of shape (N, 1)
    :param tx: input data of shape (N, D)
    :param w: model weights of shape (1, D)
    """
    pred = sigmoid(tx.dot(w))
    gradient = tx.T.dot(pred - y)
    return gradient

def compute_gradient_rlr(y, xt, w, lambda_):
    """Calculate the gradient.
    :param y: label data of shape (N, 1)
    :param tx: input data of shape (N, D)
    :param w: model weights of shape (1, D)
    :return: gradient
    """
    gradient_lr = compute_gradient_lr(y, xt, w)
    penalty = lambda_ * w
    return gradient_lr + penalty

def compute_loss_rlr(y, xt, w, lambda_):
    """compute the cost by negative log likelihood with L2
	cost = -Y'log(H) - (1 - Y')log(1 - H) + Lambda/2*W**2
	:param y: label data of shape (N, 1)
    :param tx: input data of shape (N, D)
    :param w: model weights of shape (1, D)
    :return: MSE loss for regular logistic regression
	"""
    loss_lr = compute_loss_lr(y, xt, w)
    penalty = (lambda_ / 2) * sum(w ** 2)
    return loss_lr + penalty

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.

    :param y: label data of shape (N,)
    :param tx: input data of shape (N, D)
    :param batch_size: the size of batch
	:param num_batch: the number of batches
	:param shuffle: whether or not shuffle is needed 
    :return: a batch of data	
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def mse_loss(y, tx, w):
    """Calculate the Mean Square Error loss.

    :param y: label data of shape (N,)
    :param tx: input data of shape (N, D)
    :param w: model weights of shape (1, D)
    :return: MSE loss
    """
    e = y - tx.dot(w).squeeze()
    loss = e.dot(e) / (2 * len(e))

    return loss

def mae_loss(y, tx, w):
    """Calculate the Mean Absolute Error loss.

    :param y: label data of shape (N,)
    :param tx: input data of shape (N, D)
    :param w: model weights of shape (1, D)
    :return: MAE loss
    """
    e = y - tx.dot(w).squeeze()
    loss = np.sum(np.abs(e)) / (2 * tx.shape[0])

    return loss

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    import csv
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred