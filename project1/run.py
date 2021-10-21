from helpers import data_preprocessing, get_cross_validation_data, compute_loss_rlr, \
 mse_loss, sigmoid, build_poly, split_data, mae_loss
from models import reg_logistic_regression, ridge_regression
from proj1_helpers import load_csv_data, create_csv_submission, predict_labels

import numpy as np

# set hyper-parameters
# see experiment.ipynb for parameters choose details
seed = 2021
degree = 6
lambda_ = 0.1

# load and preprocess data
y, X, _ = load_csv_data("./data/train.csv")
X = data_preprocessing(X, nan='mean')
# x_tr, x_te, y_tr, y_te = split_data(X, y, 0.8, seed=seed)
# x_tr = build_poly(x_tr, degree)
# x_te = build_poly(x_te, degree)
X = build_poly(X, degree)

# model parameter initialization
w = np.random.randn(X.shape[1], 1)

w, loss_tr = ridge_regression(y, X, mse_loss, lambda_)
# loss_te = mse_loss(y_te, x_te, w)
print(f'model training loss: {loss_tr}')
# print(f'model testing loss: {loss_te}')

_, X_test, ids = load_csv_data("./data/test.csv")
X_test = data_preprocessing(X_test, nan='mean')
X_test = build_poly(X_test, degree)
y_pred = predict_labels(w, X_test)
create_csv_submission(ids, y_pred, "submit.csv")



# model training and evaluation

# loss_te_sum = 0
# for k in range(k_fold):
#     print(f"fold {k}/{k_fold}")
#     x_tr, x_te, y_tr, y_te = get_cross_validation_data(y, x, k, degree, seed, k_fold)
#     y_tr = y_tr.reshape(-1,1)
#     y_te = y_te.reshape(-1,1)

#     # model parameter initialization
#     w = np.random.randn(x_tr.shape[1], 1)

#     w, loss_tr = reg_logistic_regression(y_tr, x_tr, lambda_, w, max_iters=max_iters, gamma=gamma)
#     loss_te = compute_loss_rlr(y_te, x_te, w, lambda_)
#     # w, loss_tr = ridge_regression(y_tr, x_tr, mse_loss, lambda_)
#     # loss_te = mse_loss(y_te, x_te, w)
#     loss_te_sum += loss_te



# y_tr = y_tr.reshape(-1,1)
# y_te = y_te.reshape(-1,1)
# w, loss_tr = reg_logistic_regression(y_tr, x_tr, lambda_, w, max_iters=max_iters, gamma=gamma)
# loss_te = compute_loss_rlr(y_te, x_te, w, lambda_)



# loss_te_avg = loss_te_sum / k_fold
# print(f'model evaluation loss: {loss_te_avg}')

# test_data = np.genfromtxt("data/test.csv", skip_header=1, delimiter = ",", dtype="str")
# x_te = data_preprocessing(test_data, nan='delete', is_test=True)
# x_te = build_poly(x_te, degree)
# # pred = sigmoid(x_te.dot(w))
# # pred = [0 if x<0.5 else 1 for x in pred]
# pred = predict_labels(w, x_te)
# create_csv_submission(ids=list(range(350000,918237+1)), y_pred=pred, name="submit.csv")