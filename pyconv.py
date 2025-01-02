import numpy as np
import pandas as pd

def diff_func(y):
    """
    Returns the first differences of a t x q matrix of data
    Python equivalent of dif.m
    
    Parameters:
    y : numpy.ndarray
        Input matrix of shape (t, q)
    
    Returns:
    numpy.ndarray
        First differences of shape (t-1, q)
    """
    return np.diff(y, axis=0)

def vec(y):
    """
    Vectorizes an (a x b) matrix y.
    Python equivalent of vec.m
    
    Parameters:
    y : numpy.ndarray
        Input matrix of shape (a, b)
    
    Returns:
    numpy.ndarray
        Vectorized matrix of shape (a*b, 1)
    """
    return y.reshape(-1, 1)
	
	
import numpy as np

def olsvarc(y, p):
    """
    Estimates a level VAR with intercept in companion format by LS
    Python equivalent of olsvarc.m
    
    Parameters:
    y : numpy.ndarray
        Input data matrix of shape (t, q)
    p : int
        Number of lags
        
    Returns:
    tuple
        (A, SIGMA, U, V, X) where:
        A: VAR coefficient matrix
        SIGMA: Covariance matrix
        U: Residuals
        V: Intercept terms
        X: Matrix of regressors
    """
    t, q = y.shape
    y = y.T  # Transpose to match MATLAB's column-major format
    
    # Create Y matrix (adjust for 0-based indexing)
    Y = y[:, (p-1):t]  # p-1 because Python is 0-based
    for i in range(1, p):
        Y = np.vstack([Y, y[:, (p-1)-i:(t)-i]])
    
    # Create X matrix with intercept (adjust for 0-based indexing)
    X = np.ones((1, t-p))
    X = np.vstack([X, Y[:, :(t-p)]])
    
    # Shift Y forward (adjust for 0-based indexing)
    Y = Y[:, 1:(t-p+1)]
    
    # Compute VAR coefficients
    A = Y @ X.T @ np.linalg.inv(X @ X.T)
    
    # Compute residuals
    U = Y - A @ X
    
    # Compute covariance matrix
    SIGMA = U @ U.T / (t - p - p*q - 1)
    
    # Extract intercept and VAR coefficients
    V = A[:, 0]
    A = A[:, 1:q*p+1]
    
    return A, SIGMA, U, V, X
	
import numpy as np

def irfvar(A, SIGMA, p, h):
    """
    Computes VAR impulse responses
    Python equivalent of irfvar.m
    
    Parameters:
    A : numpy.ndarray
        VAR coefficient matrix
    SIGMA : numpy.ndarray
        Covariance matrix
    p : int
        Number of lags
    h : int
        Horizon for impulse responses
        
    Returns:
    numpy.ndarray
        Matrix of impulse responses
    """
    # Create selection matrix J
    J = np.hstack([np.eye(3), np.zeros((3, 3*(p-1)))])
    
    # Compute Cholesky decomposition
    # Note: np.linalg.cholesky returns lower triangular, no transpose needed
    chol_sigma = np.linalg.cholesky(SIGMA)
    
    # Initialize IRF with first response (i=0)
    A0 = np.eye(A.shape[0])
    initial_resp = J @ A0 @ J.T @ chol_sigma
    IRF = initial_resp.flatten(order='F').reshape(-1, 1)
    
    # Compute responses for horizons 1 to h
    Ai = A.copy()  # Start with A^1
    for i in range(1, h+1):
        response = J @ Ai @ J.T @ chol_sigma
        IRF = np.hstack([IRF, response.flatten(order='F').reshape(-1, 1)])
        Ai = Ai @ A  # Compute next power
    
    return IRF
