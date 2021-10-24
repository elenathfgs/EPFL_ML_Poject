from helpers import data_preprocessing, get_cross_validation_data, compute_loss_rlr, \
 mse_loss, sigmoid, build_poly, split_data, mae_loss
from models import reg_logistic_regression, ridge_regression
from proj1_helpers import load_csv_data, create_csv_submission, predict_labels

import numpy as np
def train(X, y, degree, lambda_):
    """start training
    
    :param tx: input data of shape (N, D)
    :param y: label data of shape (N,)
    :param degree: degree for build_poly, expand features
    :param lambda_: weight of penalty term
    :return: w: final weight vector
    """
    seed = 2021
    x_tr, x_te, y_tr, y_te = split_data(X, y, 0.8, seed=seed)
    x_tr = build_poly(x_tr, degree)
    x_te = build_poly(x_te, degree)

    # model parameter initialization
    w = np.random.randn(x_tr.shape[1], 1)

    w, loss_tr = ridge_regression(y_tr, x_tr, mse_loss, lambda_)
 
    loss_te = mse_loss(y_te, x_te, w)
    print(f'model training loss: {loss_tr}')
    print(f'model testing loss: {loss_te}')
    return w

#load data and split it according to the column named PRI_ject_num
y, X, ids = load_csv_data("./data/train.csv") 
kind = X[:,-8]

#get index set of different PRI_ject_num(0, 1, 2&3)
zero_set = np.where(kind == 0)
one_set = np.where(kind == 1)
two_three_set = np.where((kind == 2)|(kind == 3))

#delete the columns that are meaningless or uncomputable based on specific PRI_ject_num
#collect data sets(specific row groups) according to different PRI_ject_num
zero_delete_col = [4, 5, 6, 12, 23, 24, 25, 26, 27, 28] 
one_delete_col = [4, 5, 6, 12, 26, 27, 28]

X_zero = np.delete(X, zero_delete_col, axis = 1)[zero_set,:].squeeze()

y_zero = y[zero_set]

X_one = np.delete(X, one_delete_col, axis = 1)[one_set,:].squeeze()

y_one = y[one_set]

X_two_three = X[two_three_set,:].squeeze()

y_two_three = y[two_three_set]

#train different models based on different PRI_ject_num
w0 = train(X_zero, y_zero, 10, 4e-3)

w1 = train(X_one, y_one, 10, 4e-3)

w2_3 = train(X_two_three, y_two_three, 10, 4e-3)

#make predictions based on different models and concact them
y, X_test, ids_test = load_csv_data("./data/test.csv") 

#get index set of different PRI_ject_num(0, 1, 2&3)
kind = X_test[:,-8]
zero_set = np.where(kind == 0)
one_set = np.where(kind == 1)
two_three_set = np.where((kind == 2)|(kind == 3))

#delete the columns that are meaningless or uncomputable based on specific PRI_ject_num
#collect data sets(specific row groups) according to different PRI_ject_num

X_zero = np.delete(X_test, zero_delete_col, axis = 1)[zero_set,:].squeeze()
X_zero = build_poly(X_zero, 10)
y[zero_set] = np.dot(X_zero, w0)

X_one = np.delete(X_test, one_delete_col, axis = 1)[one_set,:].squeeze()
X_one = build_poly(X_one, 10)
y[one_set] = np.dot(X_one,w1)


X_two_three = X_test[two_three_set,:].squeeze()
X_two_three = build_poly(X_two_three, 10)
y[two_three_set] = np.dot(X_two_three,w2_3)

#create labels
y[np.where(y <= 0)] = -1
y[np.where(y > 0)] = 1

create_csv_submission(ids_test, y, "submit.csv")

#test correlations 
import pandas as pd

a = pd.read_csv("0.784.csv")#onehot
b = pd.read_csv("0.796.csv")#a b corr 0.75 no onehot
d = pd.read_csv("0.798.csv")#
e = pd.read_csv("0.808.csv")#
f = pd.read_csv("0.807.csv")#
g = pd.read_csv("0.816.csv")
h = pd.read_csv("0.820.csv")
c = pd.read_csv("submit.csv")

print("a-c: corr={corr}, diff={diff}".format(corr = a["Prediction"].corr(c["Prediction"])
 ,diff = sum(a["Prediction"]==c["Prediction"])/len(a["Prediction"])))

print("b-c: corr={corr}, diff={diff}".format(corr = b["Prediction"].corr(c["Prediction"])
 ,diff = sum(b["Prediction"]==c["Prediction"])/len(a["Prediction"])))

print("d-c: corr={corr}, diff={diff}".format(corr = d["Prediction"].corr(c["Prediction"])
 ,diff = sum(d["Prediction"]==c["Prediction"])/len(a["Prediction"])))

print("e-c: corr={corr}, diff={diff}".format(corr = e["Prediction"].corr(c["Prediction"])
 ,diff = sum(e["Prediction"]==c["Prediction"])/len(a["Prediction"])))

print("f-c: corr={corr}, diff={diff}".format(corr = f["Prediction"].corr(c["Prediction"])
 ,diff = sum(f["Prediction"]==c["Prediction"])/len(a["Prediction"])))

print("g-c: corr={corr}, diff={diff}".format(corr = g["Prediction"].corr(c["Prediction"])
 ,diff = sum(g["Prediction"]==c["Prediction"])/len(a["Prediction"])))

print("h-c: corr={corr}, diff={diff}".format(corr = h["Prediction"].corr(c["Prediction"])
 ,diff = sum(h["Prediction"]==c["Prediction"])/len(a["Prediction"])))
