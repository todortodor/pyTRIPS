#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:24:24 2023

@author: slepot
"""

import matplotlib.pyplot as plt
import numpy as np

from dmsuite.poly_diff import DiffMatOnDomain, Laguerre

lag = DiffMatOnDomain(xmin=0.0, xmax=20.0, dmat=Laguerre(degree=32))
x = lag.nodes
D1 = lag.at_order(1)
D2 = lag.at_order(2)
y = np.exp(-x)
plt.plot(x, D1 @ y + y, label="error on first derivative")
plt.plot(x, D2 @ y - y, label="error on second derivative")
plt.legend()
plt.show()

#%%
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.linalg import eig
from scipy.linalg import toeplitz

def poldif(*arg):
    """
    Calculate differentiation matrices on arbitrary nodes.
      
    Returns the differentiation matrices D1, D2, .. DM corresponding to the 
    M-th derivative of the function f at arbitrarily specified nodes. The
    differentiation matrices can be computed with unit weights or 
    with specified weights.
    
    Parameters 
    ----------

    x       : ndarray
              vector of N distinct nodes
     
    M       : int 
              maximum order of the derivative, 0 < M <= N - 1    
                
    
    OR (when computing with specified weights)

    x       : ndarray
              vector of N distinct nodes
     
    alpha   : ndarray
              vector of weight values alpha(x), evaluated at x = x_j.
                
    B       : int
              matrix of size M x N, where M is the highest derivative required.  
              It should contain the quantities B[l,j] = beta_{l,j} = 
              l-th derivative of log(alpha(x)), evaluated at x = x_j.   

    Returns
    -------
   
    DM : ndarray
         M x N x N  array of differentiation matrices 
     
    Notes
    -----
    This function returns  M differentiation matrices corresponding to the 
    1st, 2nd, ... M-th derivates on arbitrary nodes specified in the array
    x. The nodes must be distinct but are, otherwise, arbitrary. The 
    matrices are constructed by differentiating N-th order Lagrange 
    interpolating polynomial that passes through the speficied points. 
    
    The M-th derivative of the grid function f is obtained by the matrix-
    vector multiplication
    
    .. math::
    
    f^{(m)}_i = D^{(m)}_{ij}f_j
     
    This function is based on code by Rex Fuzzle
    https://github.com/RexFuzzle/Python-Library
    
    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.
 
    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix 
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519
    
    """
    if len(arg) > 3:
        raise Exception('numer of arguments are either two OR three')
        
    if len(arg) == 2:
    # unit weight function : arguments are nodes and derivative order    
        x, M = arg[0], arg[1]          
        N = np.size(x); alpha = np.ones(N); B = np.zeros((M,N))
    
    # specified weight function : arguments are nodes, weights and B  matrix   
    elif len(arg) == 3:
        x, alpha, B =  arg[0], arg[1], arg[2]        
        N = np.size(x); M = B.shape[0]   
        
    I = np.eye(N)                       # identity matrix
    L = np.logical_or(I,np.zeros(N))    # logical identity matrix 
    XX = np.transpose(np.array([x,]*N))
    DX = XX-np.transpose(XX)            # DX contains entries x(k)-x(j)
    DX[L] = np.ones(N)                  # put 1's one the main diagonal
    c = alpha*np.prod(DX,1)             # quantities c(j)
    C = np.transpose(np.array([c,]*N)) 
    C = C/np.transpose(C)               # matrix with entries c(k)/c(j).    
    Z = 1/DX                            # Z contains entries 1/(x(k)-x(j)
    Z[L] = 0 #eye(N)*ZZ;                # with zeros on the diagonal.      
    X = np.transpose(np.copy(Z))        # X is same as Z', but with ...
    Xnew=X                
    
    for i in range(0,N):
        Xnew[i:N-1,i]=X[i+1:N,i]

    X=Xnew[0:N-1,:]                     # ... diagonal entries removed
    Y = np.ones([N-1,N])                # initialize Y and D matrices.
    D = np.eye(N);                      # Y is matrix of cumulative sums
    
    DM=np.empty((M,N,N))                # differentiation matrices
    
    for ell in range(1,M+1):
        Y=np.cumsum(np.vstack((B[ell-1,:], ell*(Y[0:N-1,:])*X)),0) # diags
        D=ell*Z*(C*np.transpose(np.tile(np.diag(D),(N,1))) - D)    # off-diags         
        D[L]=Y[N-1,:]
        DM[ell-1,:,:] = D 

    return DM


def lagroots(N):
    """
    Compute roots of the Laguerre polynomial of degree N
    
    Parameters
     ----------
     
    N   : int 
          degree of the Hermite polynomial
        
    Returns
    -------
    x  : ndarray
         N x 1 array of Laguerre roots
         
    """
    d0 = np.arange(1, 2*N, 2)
    d = np.arange(1, N)
    J = np.diag(d0) - np.diag(d,1) - np.diag(d,-1)

    # compute eigenvectors and eigenvalues    
    mu, v = eig(J)
    
    # return sorted, normalised eigenvalues
    return np.sort(mu)

def lagdif(N, M, b):
    '''    
    Calculate differentiation matrices using Laguerre collocation.
      
    Returns the differentiation matrices D1, D2, .. DM corresponding to the 
    M-th derivative of the function f, at the N Laguerre nodes.
        
    Parameters
    ----------
     
    N   : int 
          number of grid points
         
    M   : int
          maximum order of the derivative, 0 < M < N

    b   : float
          scale parameter, real and positive
          
    Returns
    -------
    x  : ndarray
         N x 1 array of Hermite nodes which are zeros of the N-th degree 
         Hermite polynomial, scaled by b
         
    DM : ndarray
         M x N x N  array of differentiation matrices 
        
    Notes
    -----
    This function returns  M differentiation matrices corresponding to the 
    1st, 2nd, ... M-th derivates on a Hermite grid of N points. The 
    matrices are constructed by differentiating N-th order Hermite
    interpolants. 
    
    The M-th derivative of the grid function f is obtained by the matrix-
    vector multiplication
    
    .. math::
    
    f^{(m)}_i = D^{(m)}_{ij}f_j
     
    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.
 
    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix 
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519
    
    ..[3] R. Baltensperger and M. R. Trummer, Spectral Differencing With A
    Twist, SIAM Journal on Scientific Computing 24, (2002) : 1465-1487 
           
    Examples
    --------
    
    The derivatives of functions is obtained by multiplying the vector of
    function values by the differentiation matrix. The N-point Laguerre
    approximation of the first two derivatives of y = f(x) can be obtained
    as 
    
    >>> N = 32; M = 2; b = 30
    >>> from pyddx.sc import dmsuite as dms
    >>> x, D = dms.lagdif(N, M, b)      # first two derivatives
    >>> D1 = D[0,:,:]                   # first derivative
    >>> D2 = D[1,:,:]                   # second derivative
    >>> y = np.exp(-x)                  # function at Laguerre nodes
    >>> plot(x, y, 'r', x, -D1.dot(y), 'g', x, D2.dot(y), 'b')
    >>> xlabel('$x$'), ylabel('$y$, $y^{\prime}$, $y^{\prime\prime}$')
    >>> legend(('$y$', '$y^{\prime}$', '$y^{\prime\prime}$'), loc='upper right')
    '''    
    if M >= N - 1:
        raise Exception('numer of nodes must be greater than M - 1')
        
    if M <= 0:
         raise Exception('derivative order must be at least 1')    

    # compute Laguerre nodes
    x = 0                               # include origin 
    x = np.append(x, lagroots(N-1))     # Laguerre roots
    alpha = np.exp(-x/2);               # Laguerre weights

        
    # construct beta(l,j) = d^l/dx^l (alpha(x)/alpha'(x))|x=x_j recursively
    beta = np.zeros([M , N])      
    d = np.ones(N)

    for ell in range(0, M):                           
        beta[ell,:] = pow(-0.5, ell+1)*d

    # compute differentiation matrix (b=1)
    DM = poldif(x, alpha, beta)     

    # scale nodes by the factor b
    x = x/b                 

    for ell in range(M):             
        DM[ell,:,:] = pow(b, ell+1)*DM[ell,:,:]

    return x, DM

def chebdif(N,M):
    '''    
    Calculate differentiation matrices using Chebyshev collocation.
      
    Returns the differentiation matrices D1, D2, .. DM corresponding to the 
    M-th derivative of the function f, at the N Chebyshev nodes in the 
    interval [-1,1].   
    
    Parameters
    ----------
     
    N   : int 
          number of grid points
         
    M   : int
          maximum order of the derivative, 0 < M <= N - 1

    Returns
    -------
    x  : ndarray
         N x 1 array of Chebyshev points 
         
    DM : ndarray
         M x N x N  array of differentiation matrices 
        
    Notes
    -----
    This function returns  M differentiation matrices corresponding to the 
    1st, 2nd, ... M-th derivates on a Chebyshev grid of N points. The 
    matrices are constructed by differentiating N-th order Chebyshev 
    interpolants.  
    
    The M-th derivative of the grid function f is obtained by the matrix-
    vector multiplication
    
    .. math::
    
    f^{(m)}_i = D^{(m)}_{ij}f_j
     
    The code implements two strategies for enhanced accuracy suggested by 
    W. Don and S. Solomonoff :
    
    (a) the use of trigonometric  identities to avoid the computation of
    differences x(k)-x(j) 
    
    (b) the use of the "flipping trick"  which is necessary since sin t can 
    be computed to high relative precision when t is small whereas sin (pi-t) 
    cannot.
    
    It may, in fact, be slightly better not to implement the strategies 
    (a) and (b). Please consult [3] for details.
    
    This function is based on code by Nikola Mirkov 
    http://code.google.com/p/another-chebpy

    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.
 
    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix 
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519
    
    ..[3] R. Baltensperger and M. R. Trummer, Spectral Differencing With A
    Twist, SIAM Journal on Scientific Computing 24, (2002) : 1465-1487 
           
    Examples
    --------
    
    The derivatives of functions is obtained by multiplying the vector of
    function values by the differentiation matrix. The N-point Chebyshev
    approximation of the first two derivatives of y = f(x) can be obtained
    as 
    
    >>> N = 32; M = 2; pi = np.pi
    >>> from pyddx.sc import dmsuite as dms
    >>> x, D = dms.chebdif(N, M)        # first two derivatives
    >>> D1 = D[0,:,:]                   # first derivative
    >>> D2 = D[1,:,:]                   # second derivative
    >>> y = np.sin(2*pi*x)              # function at Chebyshev nodes
    >>> plot(x, y, 'r', x, D1.dot(y), 'g', x, D2.dot(y), 'b')
    >>> xlabel('$x$'), ylabel('$y$, $y^{\prime}$, $y^{\prime\prime}$')
    >>> legend(('$y$', '$y^{\prime}$', '$y^{\prime\prime}$'), loc='upper left')
    '''

    if M >= N:
        raise Exception('numer of nodes must be greater than M')
        
    if M <= 0:
         raise Exception('derivative order must be at least 1')

    DM = np.zeros((M,N,N))
    
    n1 = int(N/2); n2 = int(round(N/2.))     # indices used for flipping trick
    k = np.arange(N)                    # compute theta vector
    th = k*np.pi/(N-1)

    # Compute the Chebyshev points

    #x = np.cos(np.pi*np.linspace(N-1,0,N)/(N-1))                # obvious way   
    x = np.sin(np.pi*((N-1)-2*np.linspace(N-1,0,N))/(2*(N-1)))   # W&R way
    x = x[::-1]
    
    # Assemble the differentiation matrices
    T = np.tile(th/2,(N,1))
    DX = 2*np.sin(T.T+T)*np.sin(T.T-T)               # trigonometric identity
    # print(n1,n2)
    DX[n1:,:] = -np.flipud(np.fliplr(DX[0:n2,:]))    # flipping trick
    DX[range(N),range(N)]=1.                         # diagonals of D
    DX=DX.T

    C = toeplitz((-1.)**k)           # matrix with entries c(k)/c(j)
    C[0,:]  *= 2
    C[-1,:] *= 2
    C[:,0] *= 0.5
    C[:,-1] *= 0.5

    Z = 1./DX                        # Z contains entries 1/(x(k)-x(j))
    Z[range(N),range(N)] = 0.        # with zeros on the diagonal.          

    D = np.eye(N)                    # D contains differentiation matrices.
                                          
    for ell in range(M):
        D = (ell+1)*Z*(C*np.tile(np.diag(D),(N,1)).T - D)      # off-diagonals    
        D[range(N),range(N)]= -np.sum(D,axis=1)        # negative sum trick
        DM[ell,:,:] = D                                # store current D in DM

    return x,DM

#%%

N = 10; M = 2; b = 1
# import dmsuite as dms
x, D = lagdif(N, M, b)      # first two derivatives
x = np.real(x)
D1 = D[0,:,:]                   # first derivative
# D1[:,-1] = D1[:,-1]-1
# D2 = D[1,:,:]                   # second derivative
y = np.exp(-x)                 # function at Laguerre nodes
plt.plot(x, y, 'r', x, D1.dot(y-y[-1]), 'g')
# plt.plot(x, y, 'r', x, D1.dot(y), 'g')
plt.xlabel('$x$'), plt.ylabel('$y$, $y^{\prime}$, $y^{\prime\prime}$')
plt.legend(('$y$', '$y^{\prime}$', '$y^{\prime\prime}$'), loc='upper right')
print(x[-1])

# #%%

# import matplotlib.pyplot as plt
# import numpy as np

# from dmsuite.poly_diff import DiffMatOnDomain, Laguerre

# lag = DiffMatOnDomain(xmin=0.0, xmax=20.0, dmat=Laguerre(degree=32))
# x = lag.nodes
# D1 = lag.at_order(1)
# # D2 = lag.at_order(2)
# y = np.exp(-x)+1
# plt.plot(x, D1 @ (y-y[-1]) + y, label="error on first derivative")
# # plt.plot(x, D2 @ y - y, label="error on second derivative")
# plt.legend()
# plt.show()