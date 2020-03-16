#!/usr/bin/env python
# encoding: utf-8
# File Name: eigen.py
# Author: Jiezhong Qiu
# Create Time: 2017/07/13 16:05
# TODO:

import argparse
import logging
from typing import Optional

import networkx as nx
import numpy as np
import scipy.io
import scipy.sparse as sparse
import scipy.sparse.linalg
import theano
from scipy.sparse import csgraph
from theano import tensor as T

logger = logging.getLogger(__name__)
theano.config.exception_verbosity = 'high'


def load_adjacency_matrix(file, variable_name="network"):
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    return data[variable_name]


def deepwalk_filter(evals, window):
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x * (1 - x ** window) / (1 - x) / window
    evals = np.maximum(evals, 0)
    logger.info("After filtering, eigenvalues in [%.3f, %.3f]", np.min(evals), np.max(evals))
    return evals


def approximate_normalized_graph_laplacian(adjacency_matrix, rank, which="LA"):
    n = adjacency_matrix.shape[0]
    logger.info("Computing Laplacian")
    L, d_rt = csgraph.laplacian(adjacency_matrix, normed=True, return_diag=True)
    # X = D^{-1/2} W D^{-1/2}
    X = sparse.identity(n) - L
    logger.info("Eigen decomposition")
    # evals, evecs = sparse.linalg.eigsh(X, rank, which=which, tol=1e-3, maxiter=300)
    evals, evecs = sparse.linalg.eigsh(X, rank, which=which)
    logger.info("Eigenvalues in [%.3f, %.3f]", np.min(evals), np.max(evals))
    logger.info("Computing D^{-1/2}U..")
    D_rt_inv = sparse.diags(d_rt ** -1)
    D_rt_invU = D_rt_inv.dot(evecs)
    return evals, D_rt_invU


def approximate_deepwalk_matrix(evals, D_rt_invU, window, vol, b):
    evals = deepwalk_filter(evals, window=window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    m = T.matrix()
    mmT = T.dot(m, m.T) * (vol / b)
    f = theano.function([m], T.log(T.maximum(mmT, 1)))
    Y = f(X.astype(theano.config.floatX))
    logger.info("Computed DeepWalk matrix with %d non-zero elements", np.count_nonzero(Y))
    return sparse.csr_matrix(Y)


def svd_deepwalk_matrix(X, dim):
    u, s, v = sparse.linalg.svds(X, dim, return_singular_vectors="u")
    # return U \Sigma^{1/2}
    return sparse.diags(np.sqrt(s)).dot(u.T).T


def netmf_large(
    adjacency_matrix,
    *,
    window: int = 10,
    rank: int = 256,
    negative: float = 1.0,
    dim: int = 128,
    output: Optional[str] = None,
):
    # perform eigen-decomposition of D^{-1/2} A D^{-1/2}
    # keep top #rank eigenpairs
    evals, D_rt_invU = approximate_normalized_graph_laplacian(adjacency_matrix, rank=rank, which="LA")

    logger.info("Approximating DeepWalk matrix with window size of %d", window)
    vol = float(adjacency_matrix.sum())
    deepwalk_matrix = approximate_deepwalk_matrix(evals, D_rt_invU, window=window, vol=vol, b=negative)

    # factorize deepwalk matrix with SVD
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=dim)

    if output is not None:
        logger.info("Save embedding to %s", output)
        np.save(output, deepwalk_embedding, allow_pickle=False)

    return deepwalk_embedding


def netmf_large_mat(matfile, *, matfile_variable_name: str = 'network', **kwargs):
    adjacency_matrix = load_adjacency_matrix(matfile, variable_name=matfile_variable_name)
    return netmf_large(adjacency_matrix, **kwargs)


def netmf_large_nx(graph: nx.Graph, **kwargs):
    logger.info('Computing adjacency matrix from NetworkX graph')
    adjacency_matrix = nx.to_numpy_matrix(graph)
    return netmf_large(adjacency_matrix, **kwargs)


def direct_compute_deepwalk_matrix(A, window, b):
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} A D^{-1/2}
    X = sparse.identity(n) - L
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        logger.info("Compute matrix %d-th power", i + 1)
        X_power = X_power.dot(X)
        S += X_power
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T)
    m = T.matrix()
    f = theano.function([m], T.log(T.maximum(m, 1)))
    Y = f(M.todense().astype(theano.config.floatX))
    return sparse.csr_matrix(Y)


def netmf_small_mat(*, matfile, matfile_variable_name: str = 'network', **kwargs):
    adjacency_matrix = load_adjacency_matrix(matfile, variable_name=matfile_variable_name)
    return netmf_small(adjacency_matrix, **kwargs)


def netmf_small(adjacency_matrix, *, window, negative, dim, output: Optional[str] = None):
    logger.info("Directly compute DeepWalk matrix with window size of %d", window)
    deepwalk_matrix = direct_compute_deepwalk_matrix(adjacency_matrix, window=window, b=negative)

    # factorize deepwalk matrix with SVD
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=dim)

    if output is not None:
        logger.info("Save embedding to %s", output)
        np.save(output, deepwalk_embedding, allow_pickle=False)

    return deepwalk_embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help=".mat input file path")
    parser.add_argument('--matfile-variable-name', default='network',
                        help='variable name of adjacency matrix inside a .mat file.')
    parser.add_argument("--output", type=str, required=True,
                        help="embedding output file path")
    parser.add_argument("--rank", default=256, type=int,
                        help="#eigenpairs used to approximate normalized graph laplacian.")
    parser.add_argument("--dim", default=128, type=int,
                        help="dimension of embedding")
    parser.add_argument("--window", default=10,
                        type=int, help="context window size")
    parser.add_argument("--negative", default=1.0, type=float,
                        help="negative sampling")
    parser.add_argument('--large', dest="large", action="store_true",
                        help="using netmf for large window size")
    parser.add_argument('--small', dest="large", action="store_false",
                        help="using netmf for small window size")
    parser.set_defaults(large=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

    if args.large:
        netmf_large_mat(
            matfile=args.input,
            matfile_variable_name=args.matfile_variable_name,
            window=args.window,
            negative=args.negative,
            dim=args.dim,
            output=args.output,
            rank=args.rank,
        )
    else:
        netmf_small_mat(
            matfile=args.input,
            matfile_variable_name=args.matfile_variable_name,
            window=args.window,
            negative=args.negative,
            dim=args.dim,
            output=args.output,
        )


if __name__ == "__main__":
    main()
