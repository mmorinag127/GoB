#
#  tsne_torch.py
#
# Implementation of t-SNE in pytorch. The implementation was tested on pytorch
# > 1.0, and it requires Numpy to read files. In order to plot the results,
# a working installation of matplotlib is required.
#
#
# The example can be run by executing: `python tsne_torch.py`
#
#
#  Created by Xiao Li on 23-03-2020.
#  Copyright (c) 2020. All rights reserved.
import numpy as np


import torch



from tqdm import tqdm


def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P


def x2p_torch(X, device, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n).to(device)
    beta = torch.ones(n, 1).to(device)
    logU = torch.log(torch.tensor([perplexity]).to(device))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    pbar_args = dict(total = n, unit = ' point', ncols = 250, disable = False, bar_format="{desc}{percentage:3.0f}%| {n_fmt: >4}/{total_fmt: >4} [{rate_fmt: >16}{postfix}]",)
    pbar_desc = f'[loop for  x2p]'

    with tqdm(**pbar_args) as pbar:
        pbar.set_description(pbar_desc)
        for i in range(n):

            # Print progress

            # Compute the Gaussian kernel and entropy for the current precision
            # there may be something wrong with this setting None
            betamin = None
            betamax = None
            Di = D[i, n_list[0:i]+n_list[i+1:n]]
            (H, thisP) = Hbeta_torch(Di, beta[i])

            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0
            while torch.abs(Hdiff) > tol and tries < 50:
                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].clone()
                    if betamax is None:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].clone()
                    if betamin is None:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                # Recompute the values
                (H, thisP) = Hbeta_torch(Di, beta[i])

                Hdiff = H - logU
                tries += 1

            # Set the final row of P
            P[i, n_list[0:i]+n_list[i+1:n]] = thisP
            pbar.set_postfix_str(f'X')
            pbar.update(1)
    
    # Return final P-matrix
    return P


def pca_torch(X, n_dim=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    (l, M) = torch.eig(torch.mm(X.t(), X), eigenvectors=True)
    #(l, M) = torch.linalg.eig(torch.mm(X.t(), X), eigenvectors=True)
    
    # split M real
    for i in range(d):
        if l[i, 1] != 0:
            M[:, i+1] = M[:, i]
            i += 1

    Y = torch.mm(X, M[:, 0:n_dim])
    return Y


def tsne(X, device, n_dim=2, initial_dim=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to n_dim dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, n_dim, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(n_dim, float):
        print("Error: array X should not have type float.")
        return -1
    if round(n_dim) != n_dim:
        print("Error: number of dimensions should be an integer.")
        return -1
    
    
    eps = torch.tensor([1e-12]).to(device)
    
    
    # Initialize variables
    X = pca_torch(X, initial_dim)
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, n_dim).to(device)
    dY = torch.zeros(n, n_dim).to(device)
    iY = torch.zeros(n, n_dim).to(device)
    gains = torch.ones(n, n_dim).to(device)

    # Compute P-values
    P = x2p_torch(X, device, 1e-5, perplexity)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.    # early exaggeration
    #print("get P shape", P.shape)
    
    P = torch.max(P, eps)
    eps = torch.tensor([1e-12]).to(device)

    

    pbar_args = dict(total = max_iter, unit = ' times', ncols = 250, disable = False, bar_format="{desc}{percentage:3.0f}%| {n_fmt: >4}/{total_fmt: >4} [{rate_fmt: >16}{postfix}]",)
    pbar_desc = f'[loop for main]'
    with tqdm(**pbar_args) as pbar:
        pbar.set_description(pbar_desc)

        # Run iterations
        for iter in range(max_iter):
            # Compute pairwise affinities
            sum_Y = torch.sum(Y*Y, 1)
            num = -2. * torch.mm(Y, Y.t())
            num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
            num[range(n), range(n)] = 0.
            num = num.to(device)
            Q = num / torch.sum(num)
            Q = Q.to(device)
            
            Q = torch.max(Q, eps)

            # Compute gradient
            PQ = P - Q
            for i in range(n):
                dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(n_dim, 1).t() * (Y[i, :] - Y), 0)

            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum

            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
            gains[gains < min_gain] = min_gain
            iY = momentum * iY - eta * (gains * dY)
            Y = Y + iY
            Y = Y - torch.mean(Y, 0)

            # Compute current value of cost function
            C = torch.sum(P * torch.log(P / Q))
            if iter % 20 == 0:
                
                pbar.set_postfix_str(f'error: {C}')
            
            # Stop lying about P-values
            if iter == 100:
                P = P / 4.
            
            pbar.update(1)

    # Return solution
    return Y

