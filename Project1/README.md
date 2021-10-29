## Machine Learning - Project 1(Team: SeaStar)

In this repository, you can find our work for the Project 1 of the [Machine Learning](https://github.com/epfml/ML_course) at [EPFL](http://epfl.ch). The background of the project could be found [here](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf.)

We take part in the [competion](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/leaderboards) and got an accuracy of 82.6%.

This file mainly focus on explaining our code.

First, ensure that you put `train.csv` and `test.csv` in the `data` folder at the root of this repository.

Then, you can run `run.py` to create `submit.csv`, the output of our model, which provides the prediction on `test.csv`.

### `helpers.py`
It contains X different major functions as following:
- **`outlier_indexs`**: return indexs of outliers of the input data
- **`remove_outliers`**: remove the outliers of the input data
- **`feature_expansion`**: Expand features of input data by applying series of different arithmetic operations 
- **`build_poly`**: Polynomial basis functions for input data x, for j=0 up to j=degree
- **`build_cross`**: Cross multiply the columns of the input data and return the result
- **`build_sqrt`**: Caculate the square root value of every elements of the input data and return the result
- **`build_log`**: Caculate the log value of every elements of the input data and return the result
- **`build_label`**: extract label from original data
- **`split_data`**: Split the dataset based on the split ratio to get train subset and test subset from original train set
- **`build_k_indices`**: build k indices for k-fold
- **`get_cross_validation_data`**: return the cross validation data
- **`load_csv_data`**: Loads data from csv file, and devide data into features and labels
- **`create_csv_submission`**: Create output csv file

### `model.py`
Contain helper methods for cross validation.
- **`least_squares_GD`**: Perform gradient descent.
- **`least_squares_SGD`**: Perform Stochastic gradient descent algorithm
- **`least_squares`**: calculate the least squares solution
- **`ridge_regression`**: execute ridge regressio
- **`logistic_regression`**: Perform logistic regression.
- **`reg_logistic_regression`**: Perform regularized logistic regression

### `loss.py`
Contain multiple methods for computing loss values:
- **`standardize`, `buid_poly`, `add_constant_column`, `na`, `impute_data` and `process_data`**: All the processing functions. See the report for explications about those functions.
- **`compute_gradient`**: Computes the gradient for gradient descent and stochastic gradient descent
- **`batch_iter`**: Generate a minibatch iterator for a dataset


### `run.py`
Contain `tain` function and multi steps of data pre-processing

