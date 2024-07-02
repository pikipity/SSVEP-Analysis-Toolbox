# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import iscomplex, ndarray

import numpy as np
import numpy.linalg as nplin
import scipy.linalg as slin

from sklearn import linear_model as sklinmodel
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
# simplefilter(action='ignore', category=ConvergenceWarning)
# @ignore_warnings(category=ConvergenceWarning)

import warnings 

from .utils import (
    eigvec, svd, get_depend_column_ind
)

from SSVEPAnalysisToolbox.utils.io import savedata

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# gpu = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)

def combine_name(method_self):
    return method_self.ID[0:-1] + '-' + method_self.LSconfig['LSMethod'] + ')'

def ssvep_lsframe(Z : ndarray,
                  Lx : ndarray,
                  Px : Optional[ndarray] = None,
                  Ly : Optional[ndarray] = None,
                  Py : Optional[ndarray] = None,
                  LSMethod: str = 'lstsq',
                  displayConvergWarn: bool = False,
                  max_n: int = 100,
                  M_error_threshold: float = 1e-6,
                  W_error_threshold: float = 1e-6,
                  smallest_n: int = 1,
                  largest_n: int = 100,
                  check_rank: bool = True,
                  l1_alpha = None,
                  l2_alpha = None,
                  alpha = None,
                  reg_iter = 1000,
                  reg_tol = 1e-4,
                  kernel_fun = None):
    if Px is None:
        _, Dx, Vx = svd(Lx, False, True)
        PKZ = Z.copy()
    else:
        if Ly is None and Py is None:
            raise ValueError("When Px is not None, Ly and Py are also should not be None.")
        try:
            LP = Ly @ Py
        except:
            LP = Ly * Py
        try:
            PKZ = LP @ Z
        except:
            PKZ = LP * Z
        _, Dx, Vx = svd(PKZ, False, True)
        PKZ = Lx @ Px @ Z
    X_rank = nplin.matrix_rank(PKZ)
    if check_rank:
        if X_rank!=min(PKZ.shape):
            max_n = smallest_n
        else:
            max_n = largest_n
    else:
        max_n = largest_n
    # max_n = largest_n
    small_value = 1e-50
    for zero_i in np.where(Dx==0)[0]:
        if zero_i == 0:
            Dx[zero_i] = small_value
        else:
            while True:
                if Dx[zero_i-1] > small_value:
                    Dx[zero_i] = small_value
                    break
                else:
                    small_value = small_value/10
    Dx_inv = nplin.inv(np.diag(Dx))
    VDV = Vx.T @ Dx_inv @ Vx
    W, M, M_error, W_error, n = lsframe(PKZ, 
                                        PKZ @ VDV, 
                                        LSMethod = LSMethod,
                                        max_n = max_n,
                                        displayConvergWarn = displayConvergWarn,
                                        M_error_threshold = M_error_threshold,
                                        W_error_threshold = W_error_threshold,
                                        l1_alpha = l1_alpha,
                                        l2_alpha = l2_alpha,
                                        alpha = alpha,
                                        reg_iter = reg_iter,
                                        reg_tol = reg_tol,
                                        kernel_fun = kernel_fun,
                                        check_full_rank = check_rank)
    if check_rank:
        if X_rank!=min(PKZ.shape):
            return W, M, M_error, W_error, n, False
        else:
            return W, M, M_error, W_error, n, True
        # W, M, M_error, W_error, n, nplin.matrix_rank(PKZ)==min(PKZ.shape)
    else:
        return W, M, M_error, W_error, n, None

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 /s2
    return s

def relu(x, thr = 0, lowest_bound = 0):
    s = np.where(x < thr, lowest_bound, x)
    return s

def binary(x, thr = 0, lowest_bound = 0):
    s = np.where(x < thr, lowest_bound, 1)
    return s

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 0)

def lsframe(X: ndarray,
            Y: ndarray,
            LSMethod: str = 'lstsq',
            displayConvergWarn: bool = False,
            max_n: int = 100,
            M_error_threshold: float = 1e-6,
            W_error_threshold: float = 1e-6,
            l1_alpha = None,
            l2_alpha = None,
            alpha = None,
            reg_iter = 1000,
            reg_tol = 1e-4,
            kernel_fun = None,
            check_full_rank = False) -> Tuple[list,list,list,list,int]:
    """
    Solve XWM'=Y

    Parameters
    ----------
    X : ndarray
    Y : ndarray

    Returns
    -------
    W : list
        W values of all iterations
    M : list
        M values of all iterations
    M_error : list
        M_error of all iterations
    W_error : list
        W_error of all iterations
    n : int
        Total iteration number
    """
    if alpha is None:
        alpha = 1
    if kernel_fun is not None:
        # Normalization
        X_norm = np.linalg.norm(X, axis = 1, keepdims=True)
        X = X / X_norm
        Y_norm = np.linalg.norm(Y, axis = 1, keepdims=True)
        Y = Y / Y_norm
        # Apply kernel
        if kernel_fun.lower() == 'sigmoid':
            X = sigmoid(X)
            Y = sigmoid(Y)
        elif kernel_fun.lower() == 'tanh':
            X = tanh(X)
            Y = tanh(Y)
        elif kernel_fun.lower() == 'relu':
            X = relu(X)
            Y = relu(Y)
        elif kernel_fun.lower() == 'binary':
            X = binary(X)
            Y = binary(Y)
        else:
            raise ValueError('Unknown kernel type')
    W = []
    M = []
    n = 0
    M_error = []
    W_error = []
    # Initialize M
    # M_init = eigvec(Y.T @ X @ nplin.inv(X.T @ X) @ X.T @ Y)
    M_init = eigvec(Y.T @ Y)
    # M_tmp = np.flip(M_tmp,1)
    M.append(M_init)
    W.append(None)
    M_error.append(np.inf)
    W_error.append(np.inf)
    while n < max_n and (M_error[-1] > M_error_threshold and W_error[-1] > W_error_threshold):
        n += 1
        # fix M and update W
        if LSMethod.lower() == 'lstsq':
            # if check_full_rank:
            #     remove_column_X = get_depend_column_ind(X)
            #     keep_column_X = np.delete(list(range(X.shape[1])), remove_column_X)
            #     YM = Y @ M[n-1]
            #     W_tmp = np.zeros((X.shape[1],YM.shape[1]))
            #     W_tmp[keep_column_X,:], _, _, _ = nplin.lstsq(X[:,keep_column_X], YM, rcond=None)
            #     if len(remove_column_X)!=0:
            #         W_tmp[remove_column_X,:], _, _, _ = nplin.lstsq(X[:,remove_column_X], YM, rcond=None)
            # else:
            #     W_tmp, _, _, _ = nplin.lstsq(X, Y @ M[n-1], rcond=None)
            try:
                W_tmp, _, _, _ = nplin.lstsq(X, Y @ M[n-1], rcond=None)
            except nplin.LinAlgError:
                # X_new = X[:,::-1]
                remove_column_X = get_depend_column_ind(X)
                keep_column_X = np.delete(list(range(X.shape[1])), remove_column_X)
                YM = Y @ M[n-1]
                W_tmp = np.zeros((X.shape[1],YM.shape[1]))
                W_tmp[keep_column_X,:], _, _, _ = nplin.lstsq(X[:,keep_column_X], YM, rcond=None)
                # W_tmp = W_tmp[::-1,:]
                # data = {"X": X,
                # "Y": Y,
                # "M": M,
                # "remove_column_X": remove_column_X,
                # "keep_column_X": keep_column_X,
                # "W_tmp": W_tmp}
                # data_file = 'test_temp.mat'
                # savedata(data_file, data, 'mat')
                # raise ValueError('Store test data when error')
        elif LSMethod.lower() == 'ols':
            # if check_full_rank:
            #     remove_column_X = get_depend_column_ind(X)
            #     keep_column_X = np.delete(list(range(X.shape[1])), remove_column_X)
            #     YM = Y @ M[n-1]
            #     W_tmp = np.zeros((X.shape[1],YM.shape[1]))
            #     W_tmp[keep_column_X,:] = ols(X[:,keep_column_X], YM)
            # else:
            #     W_tmp = ols(X, Y @ M[n-1])
            W_tmp = ols(X, Y @ M[n-1])
        elif LSMethod.lower() == 'lasso':
            if l1_alpha is None:
                raise ValueError("'l1_alpha' must be given for lasso.")
            W_tmp = lasso_sklearn(X, Y @ M[n-1], alpha * l1_alpha, 
                                  max_iter = reg_iter,
                                  tol = reg_tol)
            W_tmp = W_tmp.T
        # elif LSMethod.lower() == 'lasso-gpu':
        #     if l1_alpha is None:
        #         raise ValueError("'l1_alpha' must be given for lasso.")
        #     W_tmp, _ = ls_ElasticNet_gpu(X, Y @ M[n-1], 1,
        #                                 alpha = alpha * l1_alpha, 
        #                                 fit_intercept = False,
        #                                 max_iter = reg_iter,
        #                                 tol = reg_tol)
        elif LSMethod.lower() == 'elastic_net':
            if l1_alpha is None:
                raise ValueError("'l1_alpha' must be given for elastic_net.")
            W_tmp = elastic_net_sklearn(X, Y @ M[n-1], l1_alpha, 
                                        alpha = alpha,
                                        max_iter = reg_iter,
                                        tol = reg_tol)
            W_tmp = W_tmp.T
        elif LSMethod.lower() == 'ridge':
            if l2_alpha is None:
                raise ValueError("'l2_alpha' must be given for ridge.")
            W_tmp = ridge_sklearn(X, Y @ M[n-1], alpha * l2_alpha, 
                                  max_iter = reg_iter,
                                  tol = reg_tol)
            W_tmp = W_tmp.T
        else:
            raise ValueError("Unknown LSMethod!!")
        D_tmp = W_tmp.T @ X.T @ X @ W_tmp
        D_tmp = np.diagonal(D_tmp)
        sort_i = np.argsort(D_tmp)
        sort_i_flip = np.flip(sort_i)
        W_tmp = W_tmp[:,sort_i_flip]
        # fix W and update M
        M_tmp = np.zeros_like(M_init)
        for i in range(M_tmp.shape[1]):
            Z = Y.T @ X @ W_tmp[:,i:(i+1)]
            Uz, Dz, Vz = svd(Z, False, True)
            M_tmp[:,i:(i+1)] = Uz @ Vz
        W.append(W_tmp.copy())
        M.append(M_tmp.copy())
        # check convergence
        M_error.append(
            np.sum(np.abs(M[n]-M[n-1]))
        )
        if n==1:
            W_error.append(M_error[-1])
        else:
            W_error.append(
                np.sum(np.abs(W[n]-W[n-1]))
            )
    if n>=max_n and displayConvergWarn:
        warnings.warn("Cannot converge")
    return W, M, M_error, W_error, n

def get_lsconfig(LSconfig):
    kernel_fun = None
    if LSconfig is None:
        LSMethod = 'lstsq'
        displayConvergWarn = False
        max_n = 100
        M_error_threshold = 1e-6
        W_error_threshold = 1e-6
        l1_alpha = None
        l2_alpha = None
        alpha = None
        reg_iter = 1000
        reg_tol = 1e-4
    else:
        if len(LSconfig.values()) == 10:
            LSMethod, displayConvergWarn, max_n, M_error_threshold, W_error_threshold, l1_alpha, l2_alpha, alpha, reg_iter, reg_tol = LSconfig.values()
        elif len(LSconfig.values()) == 11:
            LSMethod, displayConvergWarn, max_n, M_error_threshold, W_error_threshold, l1_alpha, l2_alpha, alpha, reg_iter, reg_tol, kernel_fun = LSconfig.values()
        else:
            raise ValueError('The number of configures in LSconfig is wrong.')
    if LSMethod.lower() == 'lstsq' or LSMethod.lower() == 'ols':
        check_full_rank = True
    else:
        check_full_rank = False
    return LSMethod, displayConvergWarn, max_n, M_error_threshold, W_error_threshold, l1_alpha, l2_alpha, alpha, reg_iter, reg_tol, check_full_rank, kernel_fun

def canoncorr_ls(X: ndarray,
                 Y: ndarray,
                 force_output_UV: Optional[bool] = False,
                 LSconfig: Optional[dict] = None) -> Union[Tuple[ndarray, ndarray, ndarray], ndarray]:
    """
    Canonical correlation analysis based on LS framework
    """
    LSMethod, displayConvergWarn, max_n, M_error_threshold, W_error_threshold, l1_alpha, l2_alpha, alpha, reg_iter, reg_tol, check_full_rank, kernel_fun = get_lsconfig(LSconfig)
    X = X - np.mean(X,0)
    Y = Y - np.mean(Y,0)
    # Parameters of unified framework
    Z = X.copy()
    K = np.eye(X.shape[0])
    Q, _, _ = slin.qr(Y, mode = 'economic', pivoting = True)
    P = Q @ Q.T
    # LS framework for spatial filter
    _, Dx, Vx = svd(K.T @ Z, False, True)
    Dx = np.diag(Dx)
    PKZ = P.T @ K.T @ Z
    X_rank = nplin.matrix_rank(PKZ)
    if X_rank!=min(PKZ.shape) and check_full_rank:
        applied_max_n = 1
    else:
        applied_max_n = max_n
    W, _, _, _, _ = lsframe(PKZ, 
                            PKZ @ Vx.T @ nplin.inv(Dx) @ Vx, 
                            max_n = applied_max_n,
                            LSMethod = LSMethod,
                            displayConvergWarn = displayConvergWarn,
                            M_error_threshold = M_error_threshold,
                            W_error_threshold = W_error_threshold,
                            l1_alpha = l1_alpha,
                            l2_alpha = l2_alpha,
                            alpha = alpha,
                            reg_iter = reg_iter,
                            reg_tol = reg_tol,
                            kernel_fun = kernel_fun,
                            check_full_rank = check_full_rank)
    W = W[-1].copy()
    # Weights of reference signal and correlation coefficients
    V, _, _, _ = nplin.lstsq(Y, X @ W, rcond=None)
    if X_rank!=min(PKZ.shape) and check_full_rank:
        W, _, _, _ = nplin.lstsq(Z, Y @ V, rcond=None)
    r = []
    for i in range(W.shape[1]):
        r.append(
            np.corrcoef(W[:,i:(i+1)].T @ X.T, V[:,i:(i+1)].T @ Y.T)[0,1]
        )
    r = np.array(r)
    # return results
    if force_output_UV:
        return W, V, r
    else:
        return r

def ols(X, y, fit_intercept=False):
    """Ordinary Least Squares (OLS) Regression model with intercept term.
    Fits an OLS regression model using the closed-form OLS estimator equation.
    Intercept term is included via design matrix augmentation.
    Params:
        X - NumPy matrix, size (N, p), of numerical predictors
        y - NumPy array, length N, of numerical response
        fit_intercept - Boolean indicating whether to include an intercept term
    Returns:
        NumPy array, length p + 1, of fitted model coefficients
    """
    if fit_intercept:
        X = np.hstack((np.ones((X.shape[0], 1)), X))
    _, n_y = y.shape
    w = []
    for i in range(n_y):
        w.append(nplin.solve(np.dot(X.T, X), np.dot(X.T, y[:,i])))
    w = [np.expand_dims(tmp,axis = 1) for tmp in w]
    w = np.concatenate(w, axis = 1)
    return w

def lasso_sklearn(X, y, l1, 
                  fit_intercept = False,
                  max_iter = 1000,
                  tol = 1e-4):
    clf = sklinmodel.Lasso(alpha = l1,
                           fit_intercept = fit_intercept,
                           max_iter = max_iter,
                           tol = tol,
                           precompute = True)
    # alphas, clf, dual_gaps = sklinmodel.lasso_path(X, y, eps = 1e-5)
    # clf = clf[0,:,:]
    simplefilter(action='ignore', category=ConvergenceWarning)
    clf.fit(X, y)
    if fit_intercept:
        return np.concatenate(clf.coef_, np.expand_dims(clf.intercept_, axis = 1), axis = 0)
    return clf.coef_

def lasso(X, y, l1, tol=1e-6, path_length=100, return_path=False, fit_intercept=False):
    """The Lasso Regression model with intercept term.
    Intercept term included via design matrix augmentation.
    Pathwise coordinate descent with co-variance updates is applied.
    Path from max value of the L1 tuning parameter to input tuning parameter value.
    Features must be standardized (centered and scaled to unit variance)
    Params:
        X - NumPy matrix, size (N, p), of standardized numerical predictors
        y - NumPy array, length N, of numerical response
        l1 - L1 penalty tuning parameter (positive scalar)
        tol - Coordinate Descent convergence tolerance (exited if change < tol)
        path_length - Number of tuning parameter values to include in path (positive integer)
        return_path - Boolean indicating whether model coefficients along path should be returned
    Returns:
        if return_path == False:
            NumPy array, length p + 1, of fitted model coefficients
        if return_path == True:
            List, length 3, of last fitted model coefficients, tuning parameter path and coefficient values
    Reference:
        https://towardsdatascience.com/regularized-linear-regression-models-dcf5aa662ab9
    """
    if fit_intercept:
        X = np.hstack((np.ones((len(X), 1)), X))
    m, n = np.shape(X)
    B_star = np.zeros((n))
    l_max = max(list(abs(np.dot(np.transpose(X[:, 1:]), y)))) / m
    # At or above l_max, all coefficients (except intercept) will be brought to 0
    if l1 >= l_max:
        if fit_intercept:
            B_star = np.append(np.mean(y), np.zeros((n - 1)))
        else:
            B_star = np.zeros(n)
        if return_path:
            return [B_star, None, None]
        else:
            return B_star
    l_path = np.geomspace(l_max, l1, path_length)
    coeffiecients = np.zeros((len(l_path), n))
    for i in range(len(l_path)):
        while True:
            B_s = B_star
            for j in range(n):
                k = np.where(B_s != 0)[0]
                update = (1/m)*((np.dot(X[:,j], y)- \
                                np.dot(np.dot(X[:,j], X[:,k]), B_s[k]))) + \
                                B_s[j]
                B_star[j] = (np.sign(update) * max(abs(update) - l_path[i], 0))
            if np.all(abs(B_s - B_star) < tol):
                coeffiecients[i, :] = B_star
                break
    if return_path:
        return [B_star, l_path, coeffiecients]
    else:
        return B_star

def elastic_net_sklearn(X, y, l1_ratio,
                        alpha = 1, 
                        fit_intercept = False,
                        max_iter = 1000,
                        tol = 1e-4):
    clf = sklinmodel.ElasticNet(alpha = alpha,
                                l1_ratio = l1_ratio,
                                fit_intercept = fit_intercept,
                                max_iter = max_iter,
                                tol = tol,
                                precompute = True)
    simplefilter(action='ignore', category=ConvergenceWarning)
    clf.fit(X, y)
    if fit_intercept:
        return np.concatenate(clf.coef_, np.expand_dims(clf.intercept_, axis = 1), axis = 0)
    return clf.coef_

def elastic_net(X, y, l, alpha, tol=1e-4, path_length=100, return_path=False, fit_intercept=False):
    """The Elastic Net Regression model with intercept term.
    Intercept term included via design matrix augmentation.
    Pathwise coordinate descent with co-variance updates is applied.
    Path from max value of the L1 tuning parameter to input tuning parameter value.
    Features must be standardized (centered and scaled to unit variance)
    Params:
        X - NumPy matrix, size (N, p), of standardized numerical predictors
        y - NumPy array, length N, of numerical response
        l - l penalty tuning parameter (positive scalar)
        alpha - alpha penalty tuning parameter (positive scalar between 0 and 1)
        tol - Coordinate Descent convergence tolerance (exited if change < tol)
        path_length - Number of tuning parameter values to include in path (positive integer)
    Returns:
        NumPy array, length p + 1, of fitted model coefficients
    Reference:
        https://towardsdatascience.com/regularized-linear-regression-models-dcf5aa662ab9
    """
    if fit_intercept:
        X = np.hstack((np.ones((len(X), 1)), X))
    m, n = np.shape(X)
    B_star = np.zeros((n))
    if alpha == 0:
        l2 = 1e-15
    l_max = max(list(abs(np.dot(np.transpose(X), y)))) / m / alpha
    if l >= l_max:
        if fit_intercept:
            B_star = np.append(np.mean(y), np.zeros((n - 1)))
        else:
            B_star = np.zeros(n)
        if return_path:
            return [B_star, None, None]
        else:
            return B_star
    l_path = np.geomspace(l_max, l, path_length)
    coeffiecients = np.zeros((len(l_path), n))
    for i in range(path_length):
        while True:
            B_s = B_star
            for j in range(n):
                k = np.where(B_s != 0)[0]
                update = (1/m)*((np.dot(X[:,j], y)- \
                                np.dot(np.dot(X[:,j], X[:,k]), B_s[k]))) + \
                                B_s[j]
                B_star[j] = (np.sign(update) * max(
                    abs(update) - l_path[i] * alpha, 0)) / (1 + (l_path[i] * (1 - alpha)))
            if np.all(abs(B_s - B_star) < tol):
                coeffiecients[i, :] = B_star
                break
    if return_path:
        return [B_star, l_path, coeffiecients]
    else:
        return B_star

def ridge_sklearn(X, y, l2, 
                 fit_intercept = False,
                 max_iter = 1000,
                 tol = 1e-4):
    clf = sklinmodel.Ridge(alpha = l2,
                           fit_intercept = fit_intercept,
                           max_iter = max_iter,
                           tol = tol)
    simplefilter(action='ignore', category=ConvergenceWarning)
    clf.fit(X, y)
    if fit_intercept:
        return np.concatenate(clf.coef_, np.expand_dims(clf.intercept_, axis = 1), axis = 0)
    return clf.coef_

def ridge(X, y, l2, fit_intercept=False):
    """Ridge Regression model with intercept term.
    L2 penalty and intercept term included via design matrix augmentation.
    This augmentation allows for the OLS estimator to be used for fitting.
    Params:
        X - NumPy matrix, size (N, p), of numerical predictors
        y - NumPy array, length N, of numerical response
        l2 - L2 penalty tuning parameter (positive scalar) 
    Returns:
        NumPy array, length p + 1, of fitted model coefficients
    Reference:
        https://towardsdatascience.com/regularized-linear-regression-models-44572e79a1b5
    """
    if fit_intercept:
        m, n = np.shape(X)
        upper_half = np.hstack((np.ones((m, 1)), X))
    else:
        upper_half = X
    lower = np.zeros((n, n))
    np.fill_diagonal(lower, np.sqrt(l2))
    lower_half = np.hstack((np.zeros((n, 1)), lower))
    X = np.vstack((upper_half, lower_half))
    y = np.append(y, np.zeros(n))
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

# def ls_ElasticNet_gpu(X, Y, l1_ratio,
#                       alpha = 1, 
#                       fit_intercept = False,
#                       max_iter = 1000,
#                       tol = 1e-4,
#                       lr = 1e-3):
#     n_samples, n_input = X.shape
#     n_epoch = max_iter
#     lam = alpha
#     n_component = Y.shape[1]
#     #
#     x = tf.compat.v1.placeholder(dtype = tf.float64, shape = [None, n_input])
#     y = tf.compat.v1.placeholder(dtype = tf.float64, shape = [None, n_component])
#     w = tf.Variable(np.random.rand(n_input, n_component))
#     b = tf.Variable(np.random.rand(1, n_component))
#     if fit_intercept:
#         z = x @ w + b
#     else:
#         z = x @ w
#     #
#     mse_loss = tf.reduce_mean((z - y) ** 2, axis = 0)
#     l1_loss = lam * l1_ratio * tf.reduce_sum(tf.abs(w), axis = 0)
#     l2_loss = lam * (1 - l1_ratio) * tf.reduce_sum(w ** 2, axis = 0)
#     if l1_ratio == 1:
#         loss = mse_loss + l1_loss
#     elif l1_ratio == 0:
#         loss = mse_loss + l2_loss
#     else:
#         loss = mse_loss + l1_loss + l2_loss
#     loss = tf.reduce_sum(loss)
#     op = tf.compat.v1.train.AdamOptimizer(lr).minimize(loss)
#     #
#     W_final = []
#     B_final = []
#     losses_final = []
#     for i in range(1):
#         losses = []
#         with tf.compat.v1.Session() as sess:
#             sess.run(tf.compat.v1.global_variables_initializer())
#             for e in range(n_epoch):
#                 _, loss_ = sess.run([op, loss], feed_dict={x: X, y: Y})
#                 losses.append(loss_)
#                 if loss_ < tol:
#                     break
#                 if e > 0:
#                     losses_diff = abs(losses[-1] - losses[-2])
#                     if losses_diff < 1e-6:
#                         break
#             w_res = sess.run(w)
#             b_res = sess.run(b)
#         losses_final.append(losses)
#         W_final.append(w_res)
#         B_final.append(b_res)
#     W_final = np.concatenate(W_final, axis = 1)
#     B_final = np.concatenate(B_final, axis = 1)
#     if fit_intercept:
#         W_final = np.concatenate((W_final, B_final), axis = 0)
#     return W_final, losses_final
