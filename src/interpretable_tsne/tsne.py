# This is a modification of the tSNE implementation from sklearn:
# https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/manifold/_t_sne.py
# Original copyright info:
# Author: Alexander Fabisch  -- <afabisch@informatik.uni-bremen.de>
# Author: Christopher Moody <chrisemoody@gmail.com>
# Author: Nick Travers <nickt@squareup.com>
# License: BSD 3 clause (C) 2014
#
# Author: Matthew Scicluna -- <mattcscicluna@gmail.com>
# Licence: BSD 3 clause (C) 2023

from time import time
import sys

import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.validation import check_non_negative
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

from ._bintree import _binary_search_perplexity
from ._barnes_hut_tsne import gradient
from ._grad_comps import compute_p_gradient, compute_attr

MACHINE_EPSILON = np.finfo(np.double).eps
#sys.tracebacklimit = 0


def _joint_probabilities(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances.
    Parameters
    ----------
    distances : array, shape (n_samples * (n_samples-1) / 2,)
        Distances of samples are stored as condensed matrices, i.e.
        we omit the diagonal and duplicate entries and store everything
        in a one-dimensional array.
    desired_perplexity : float
        Desired perplexity of the joint probability distributions.
    verbose : int
        Verbosity level.
    Returns
    -------
    P : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    conditional_P : np.array, shape (n_samples, n_samples)
        conditional probability matrix.
    betas: np.array, shape (n_samples)
        beta terms
    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances = distances.astype(np.float32, copy=False)
    conditional_P, betas = _binary_search_perplexity(
        distances, desired_perplexity, verbose)
    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    return P, conditional_P, betas


def _joint_probabilities_nn(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances using just nearest
    neighbors.
    This method is approximately equal to _joint_probabilities. The latter
    is O(N), but limiting the joint probability to nearest neighbors improves
    this substantially to O(uN).
    Parameters
    ----------
    distances : CSR sparse matrix, shape (n_samples, n_samples)
        Distances of samples to its n_neighbors nearest neighbors. All other
        distances are left to zero (and are not materialized in memory).
    desired_perplexity : float
        Desired perplexity of the joint probability distributions.
    verbose : int
        Verbosity level.
    Returns
    -------
    P : csr sparse matrix, shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors.
    conditional_P : csr sparse matrix, shape (n_samples, n_samples)
        Condensed conditional probability matrix with only nearest neighbors.
    betas: np.array, shape (n_samples)
        beta terms
    """
    t0 = time()
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)
    conditional_P, betas = _binary_search_perplexity(
        distances_data, desired_perplexity, verbose)
    assert np.all(np.isfinite(conditional_P)), \
        "All probabilities should be finite"

    conditional_P = csr_matrix((conditional_P.ravel(), distances.indices,
                                distances.indptr),
                               shape=(n_samples, n_samples))

    # Symmetrize the joint probability distribution using sparse operations
    P = conditional_P + conditional_P.T

    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    if verbose >= 2:
        duration = time() - t0
        print("[t-SNE] Computed conditional probabilities in {:.3f}s"
              .format(duration))

    assert np.all(betas < 1000), \
    "Beta terms have numerically unstable values! Check your data!"

    return P, conditional_P, betas


def _compute_attr_bh(Y,
                     qt,
                     sumQ,
                     dYx,
                     dP,
                     P,
                     degrees_of_freedom,
                     n_samples,
                     n_components,
                     angle,
                     verbose,
                     num_threads):
    """Compute attribution using Barnes-Hut approximation

    Parameters
    ----------
    Y : array, shape (n_samples, n_components)
        embedding.
    qt: quadtree
        quadtree object
    dYx: array, shape (n_components, n_samples, n_dimensions)
        Gradients at previous timesteps
    dP: array, shape (n_samples * (n_samples-1) / 2, n_dimensions)
        array of gradients of P w.r.t. x. Each row index corresponds to a 
        non-zero entry of P (which can be retrieved by P.indices and P.indptr)
    P : csr sparse matrix, shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors.
    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.
    n_samples : int
        Number of samples.
    n_components : int
        Dimension of the embedded space.
    angle: float
        See `compute_attr`
    verbose: int
        See `compute_attr`
    num_threads: int
        see `compute_attr`

    Returns
    -------
    new_dYx: array, shape (n_components, n_samples, n_dimensions)
        Gradients at current timestep
    """

    dP_idxs = P.indices.astype(np.int64, copy=False)  # dP_ij = 0 iff P_ij = 0!
    dP_idptr = P.indptr.astype(np.int64, copy=False)
    P = P.data.astype(np.float32, copy=False)

    new_dYx = compute_attr(Y.reshape(n_samples, n_components),
                           qt,
                           sumQ,
                           dYx,
                           dP,
                           dP_idxs,
                           dP_idptr,
                           P,
                           angle,
                           degrees_of_freedom,
                           verbose,
                           num_threads)

    # c term does not seem to be included in gradient anyways!
    #c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    #new_dYx *= c  # times by 4 in usual scenerio

    return new_dYx


def _compute_attr(Y,
                  qt,
                  sumQ,
                  dYx,
                  dP,
                  P,
                  degrees_of_freedom,
                  n_samples,
                  n_components,
                  angle,
                  verbose,
                  num_threads):
    # computes Phi
    Y = Y.reshape(n_samples, n_components)
    D = 1 / (1 + pairwise_distances(Y)**2)
    phi = (np.expand_dims(Y, 1) - np.expand_dims(Y, 0)) * np.expand_dims(D, 2)  # (N, N, 2)

    # computes dP/dx * phi
    # (2, N, N, d) -> (2, N, d)
    pos_array = (np.expand_dims(phi.transpose(2,0,1), 3) * np.expand_dims(dP, 0)).sum(2)

    # computes dPhi/dy
    dPhiy = (np.expand_dims(D, axis=(2,3)) * np.expand_dims(np.eye(phi.shape[2], phi.shape[2]), axis=(0,1)))-2*np.expand_dims(phi, 2)*np.expand_dims(phi, 3)

    # computes dPhi/dx
    # (N, N, 1, 2, 2) @ (N, 1, d, 1, 2) -> (N, N, d, 2)
    dPhix = (np.expand_dims(dPhiy, 2) @ np.expand_dims(dYx.transpose(1, 2, 0), axis=(1, 4))).squeeze()

    #(num.unsqueeze(2).unsqueeze(2)*(torch.eye(phi.shape[2]).unsqueeze(0).unsqueeze(0).to(device))-2*(phi.unsqueeze(2)*phi.unsqueeze(3)))
    #DPhix = (DPhiy.unsqueeze(2) @ prev_DYx.permute(1,2,0).unsqueeze(1).unsqueeze(4)).squeeze()

    # computes (dP/dx * phi + dPhi/dx * P) [this is the positive half of the gradient]
    # (2, N, N, d) x (1, N, N, 1) -> (2, N, d)
    pos_array += (dPhix.transpose(3, 0, 1, 2) * np.expand_dims(squareform(P), axis=(0,3))).sum(2)

    # computes negative attr
    #Q, D, Ydiff, S = _compute_q_phi_debug(Y, qt, 0.5, sumQ, 1)
    phi = phi.transpose(2,0,1)

    Q = D / (D - np.eye(D.shape[0])).sum()
    dPhixQ = ((dPhix.transpose(3, 0, 1, 2) * np.expand_dims(Q, axis=(0,3)))).sum(2)

    EQy = (np.expand_dims(Q, 0) * phi).sum(2)  # (2, N)
    dQy = -2*Q*(phi-2*np.expand_dims(EQy, 1))  # (2, N, N)
    dQx = (np.expand_dims(dYx, 2) * np.expand_dims(dQy, 3)).sum(0)
    dQxPhi = ((np.expand_dims(dQx, 0) * np.expand_dims(phi, 3))).sum(2)

    neg_array = dQxPhi + dPhixQ

    # Combine them
    ddYx = pos_array - neg_array

    return ddYx, pos_array, neg_array


def _compute_dp_bh(X, P_ji, P, betas, degrees_of_freedom, num_threads):
    """
    Computes dP with Barnes-Hut approx
    """
    val_P_ji = P_ji.data.astype(np.float32, copy=False)
    P_ji_idxs = P_ji.indices.astype(np.int64, copy=False)
    P_ji_idptr = P_ji.indptr.astype(np.int64, copy=False)
    P_ij = P_ji.T.tocsr()
    val_P_ij = P_ij.data.astype(np.float32, copy=False)
    P_ij_idxs = P_ij.indices.astype(np.int64, copy=False)
    P_ij_idptr = P_ij.indptr.astype(np.int64, copy=False)
    dP_idxs = P.indices.astype(np.int64, copy=False)  # dP_ij = 0 iff P_ij = 0!
    dP_idptr = P.indptr.astype(np.int64, copy=False)

    EX, dP = compute_p_gradient(X,
                                P_ij_idxs,
                                P_ij_idptr,
                                val_P_ij,
                                P_ji_idxs,
                                P_ji_idptr,
                                val_P_ji,
                                dP_idxs,
                                dP_idptr,
                                betas,
                                num_threads)

    return EX, dP.astype(np.float32, copy=False)  # ensure float32


def _compute_dp(X, conditional_P, P, betas):
    """
    Computes dP exactly
    """
    _P_ji = np.expand_dims(conditional_P, 2)
    _P_ij = np.expand_dims(conditional_P.T, 2)
    dij = -((np.expand_dims(X, 1)-np.expand_dims(X, 0)))*np.expand_dims(np.expand_dims(betas, 1), 1)
    dji = ((np.expand_dims(X, 1)-np.expand_dims(X, 0)))*np.expand_dims(np.expand_dims(betas, 1), 0)
    dP_ji = _P_ji*(dij - np.expand_dims((dij*_P_ji).sum(1), 1))
    dP_ij = dji*_P_ij*(1-_P_ij)
    dP = (dP_ij + dP_ji)/(2*X.shape[0])

    EX = (_P_ji*dij).sum(1)

    return EX, dP.astype(np.float32, copy=False)


def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components,
                   skip_num_points=0, compute_error=True):
    """t-SNE objective function: gradient of the KL divergence
    of p_ijs and q_ijs and the absolute error.
    Parameters
    ----------
    params : array, shape (n_params,)
        Unraveled embedding.
    P : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.
    n_samples : int
        Number of samples.
    n_components : int
        Dimension of the embedded space.
    skip_num_points : int (optional, default:0)
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.
    compute_error: bool (optional, default:True)
        If False, the kl_divergence is not computed and returns NaN.
    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.
    grad : array, shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    qt: None
        No quadtree object used!
    """
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if compute_error:
        kl_divergence = 2.0 * np.dot(
            P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    # Gradient: dC/dY
    # pdist always returns double precision distances. Thus we need to take
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    qt = None  # no quadtree used!
    sumQ = None  # no need to compute this!

    return kl_divergence, grad, qt, sumQ


def _kl_divergence_bh(params, P, degrees_of_freedom, n_samples, n_components,
                      angle=0.5, skip_num_points=0, verbose=False,
                      compute_error=True, num_threads=1):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.
    Uses Barnes-Hut tree methods to calculate the gradient that
    runs in O(NlogN) instead of O(N^2)
    Parameters
    ----------
    params : array, shape (n_params,)
        Unraveled embedding.
    P : csr sparse matrix, shape (n_samples, n_sample)
        Sparse approximate joint probability matrix, computed only for the
        k nearest-neighbors and symmetrized.
    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.
    n_samples : int
        Number of samples.
    n_components : int
        Dimension of the embedded space.
    angle : float (default: 0.5)
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.
    skip_num_points : int (optional, default:0)
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.
    verbose : int
        Verbosity level.
    compute_error: bool (optional, default:True)
        If False, the kl_divergence is not computed and returns NaN.
    num_threads : int (optional, default:1)
        Number of threads used to compute the gradient. This is set here to
        avoid calling _openmp_effective_n_threads for each gradient step.
    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.
    grad : array, shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    qt: quadtree
        quadtree object.
    """
    params = params.astype(np.float32, copy=False)
    X_embedded = params.reshape(n_samples, n_components)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)

    grad = np.zeros(X_embedded.shape, dtype=np.float32)

    error, qt, sumQ = gradient(val_P,
                               X_embedded,
                               neighbors,
                               indptr,
                               grad,
                               angle,
                               n_components,
                               verbose,
                               dof=degrees_of_freedom,
                               compute_error=compute_error,
                               num_threads=num_threads)

    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c

    return error, grad, qt, sumQ


def resolve_attr_value(attr_style, curr_attr, dYx, p, pos_array, neg_array, grad, accumulate, at_checkpoint, percentile_ceil=99.5):

    # accumulating attrs need to always be computed
    if accumulate:
        if attr_style == 'mean_grad_xy':
            curr_attr += dYx
        elif attr_style == 'mean_grad_norm':
            curr_attr += np.expand_dims((2*np.expand_dims(p.reshape(dYx.shape[1], -1).T, 2)*dYx).sum(0), 0)
        elif attr_style == 'kl_obj_mean':
            #curr_attr += np.abs(np.expand_dims((2*np.expand_dims(grad.reshape(dYx.shape[1], -1).T, 2)*dYx).sum(0), 0))  # With abs value
            curr_attr += np.expand_dims((2*np.expand_dims(grad.reshape(dYx.shape[1], -1).T, 2)*dYx).sum(0), 0)  # Without abs value

    # non-accumulating attrs only need to be computed at checkpoints
    if at_checkpoint:
        if attr_style == 'grad_xy':
            curr_attr = dYx
        elif attr_style == 'grad_norm':
            curr_attr = np.expand_dims((2*np.expand_dims(p.reshape(dYx.shape[1], -1).T, 2)*dYx).sum(0), 0)
        elif attr_style == 'pos_neg_force':
            curr_attr = np.stack([(2*np.expand_dims(p.reshape(dYx.shape[1], -1).T, 2)*pos_array).sum(0),
                                  (2*np.expand_dims(p.reshape(dYx.shape[1], -1).T, 2)*neg_array).sum(0)])
        elif attr_style == 'kl_obj':
            #curr_attr = np.stack([(2*np.expand_dims(grad.reshape(dYx.shape[1], -1).T, 2)*pos_array).sum(0),
            #                      (2*np.expand_dims(grad.reshape(dYx.shape[1], -1).T, 2)*neg_array).sum(0)])
            curr_attr = np.expand_dims((2*np.expand_dims(grad.reshape(dYx.shape[1], -1).T, 2)*dYx).sum(0), 0)

    return curr_attr


def _gradient_descent(objective, attr_function, p0, dP, dYx, diYx, attr_style, checkpoint_every, return_obj, it, n_iter,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7, verbose=0, args=None, kwargs=None, attr_args=None):
    """Batch gradient descent with momentum and individual gains.
    Parameters
    ----------
    objective : function or callable
        Should return a tuple of cost and gradient for a given parameter
        vector. When expensive to compute, the cost can optionally
        be None and can be computed every n_iter_check steps using
        the objective_error function.
    attr_function: function or callable
        Should return attributions at time step
    p0 : array-like, shape (n_params,)
        Initial parameter vector.
    dP : numpy.ndarray, shape (n_nonzero_p, n_features)
    dYx :  numpy.ndarray, shape (n_components, n_samples, n_features)
           array of current gradients of embeddings w.r.t. data inputs
    diYx : numpy.ndarray, shape (n_components, n_samples, n_features)
           intermediate values needed to compute gradients of current 
           step given gradients of previous one
    attr_style: string
                what attribution to compute.
                Options:
                    * 'none' : does not compute any attributions (so runs usual tSNE).
                    * 'grad_xy' : returns gradients w.r.t. tSNE components
                    * 'grad_norm' : returns gradients w.r.t. norm of tSNE components
                    * 'mean_grad_xy' : returns mean of gradients (over all steps)
                    * 'mean_grad_norm' : returns mean of norm of gradients (over all steps)
                    * 'pos_neg_force': +ve and -ve forces (experimental)
                    * 'kl_obj': returns gradients w.r.t KL objective
                    * 'kl_obj_mean': returns mean of gradients  w.r.t KL objective (over all steps)
    return_obj: dict
        dict of objects to return
        (list of attributions and embeddings at select time points)
    checkpoint_every: list
        when to return attributions
    it : int
        Current number of iterations (this function will be called more than
        once during the optimization).
    n_iter : int
        Maximum number of gradient descent iterations.
    n_iter_check : int
        Number of iterations before evaluating the global error. If the error
        is sufficiently low, we abort the optimization.
    n_iter_without_progress : int, optional (default: 300)
        Maximum number of iterations without progress before we abort the
        optimization.
    momentum : float, within (0.0, 1.0), optional (default: 0.8)
        The momentum generates a weight for previous gradients that decays
        exponentially.
    learning_rate : float, optional (default: 200.0)
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers.
    min_gain : float, optional (default: 0.01)
        Minimum individual gain for each parameter.
    min_grad_norm : float, optional (default: 1e-7)
        If the gradient norm is below this threshold, the optimization will
        be aborted.
    verbose : int, optional (default: 0)
        Verbosity level.
    args : sequence
        Arguments to pass to objective function.
    kwargs : dict
        Keyword arguments to pass to objective function.
    Returns
    -------
    p : array, shape (n_params,)
        Optimum parameters.
    error : float
        Optimum.
    i : int
        Last iteration.
    dYx :  numpy.ndarray, shape (n_components, n_samples, n_features)
        array of current gradients of embeddings w.r.t. data inputs
    diYx : numpy.ndarray, shape (n_components, n_samples, n_features)
        intermediate values needed to compute gradients of current
        step given gradients of previous one
    return_obj : dict
        attributions and embeddings to return
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    # initialize running average
    attr = None
    if (attr_style == 'mean_grad_xy') or (attr_style == 'mean_grad_norm') or (attr_style == 'kl_obj_mean'):
        if len(return_obj['attrs']) > 0:
            attr = return_obj['attrs'][-1]
        else:
            if (attr_style == 'mean_grad_xy'):
                attr = np.zeros_like(dYx)
            else:
                # tSNE component dimension is 1!
                attr = np.zeros(shape=[1, dYx.shape[1], dYx.shape[2]])

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(float).max
    best_error = np.finfo(float).max
    best_iter = i = it

    tic = time()

    for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs['compute_error'] = check_convergence or i == n_iter - 1

        error, grad, qt, sumQ = objective(p, *args, **kwargs)

        # Compute dQ, dPhi, Phi
        if attr_style != 'none':
            ddYx, pos_array, neg_array = attr_function(p, qt, sumQ, dYx, dP, *args, **attr_args)

            # clip gradient updates at each step to avoid numeric instability
            np.clip(ddYx, -1, 1, out=ddYx)
        else:
            ddYx, pos_array, neg_array = None, None, None
        
        grad_norm = linalg.norm(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if attr_style != 'none':
            diYx = momentum * diYx - learning_rate * np.expand_dims(gains.reshape(ddYx.shape[0],-1), 2) * ddYx
            dYx = dYx + diYx   # need to update dYx to new value!

            if ((i % 25) == 0) and (verbose >= 3):
                print('iter: {}'.format(i))
                #i1, i2 = np.where(p.reshape(-1,2) == p.max())
                #print(i1[0], i2[0])
                #print(p.reshape(-1, 2)[i1[0]-2:i1[0]+2,:])

                print("nans in locations (count): {}".format(np.unique(np.where(np.isnan(dYx))[1]).shape[0]))
                print("dYx. max: {} min: {} median: {}".format(np.nanmax(dYx), np.nanmin(dYx), np.nanmedian(dYx)))
                print("diYx. max: {} min: {} median: {}".format(np.nanmax(diYx), np.nanmin(diYx), np.nanmedian(diYx)))
                print("ddYx. max: {} min: {} median: {}".format(np.nanmax(ddYx), np.nanmin(ddYx), np.nanmedian(ddYx)))
                print("pos. max: {} min: {} median: {}".format(np.nanmax(pos_array), np.nanmin(pos_array), np.nanmedian(pos_array)))
                print("neg. max: {} min: {} median: {}".format(np.nanmax(neg_array), np.nanmin(neg_array), np.nanmedian(neg_array)))
                #print(np.unique(np.argwhere(dYx > 1)[:,1]))
                #print(np.unique(np.argwhere(diYx > 1)[:,1]))
                #print(np.unique(np.argwhere(ddYx > 1)[:,1]))
                #print(np.unique(np.argwhere(pos_array > 1)[:,1]))
                #print(np.unique(np.argwhere(neg_array > 1)[:,1]))

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break

        if len(checkpoint_every) > 0:
            if i == checkpoint_every[0]:

                attr = resolve_attr_value(attr_style, attr, dYx, p, pos_array, neg_array, grad, accumulate=True, at_checkpoint=True)
                checkpoint_every.pop(0)
                if attr_style != 'none':
                    return_obj['attrs'].append(attr.copy())
                #return_obj['embeddings'].append(p.copy().reshape([dYx.shape[1], -1]))
                return_obj['embeddings'].append(p.copy().reshape([-1, 2]))  # TODO: make this better
        else:
            attr = resolve_attr_value(attr_style, attr, dYx, p, pos_array, neg_array, grad, accumulate=True, at_checkpoint=False)

    # Add at the end
    attr = resolve_attr_value(attr_style, attr, dYx, p, pos_array, neg_array, grad, accumulate=False, at_checkpoint=True)

    #  always add last attr and embedding
    if attr_style != 'none':
        return_obj['attrs'].append(attr)
    #return_obj['embeddings'].append(p.reshape([dYx.shape[1], -1]))
    return_obj['embeddings'].append(p.copy().reshape([-1, 2]))  # TODO: make this better

    return p, error, i, dYx, diYx, return_obj


@_deprecate_positional_args
def trustworthiness(X, X_embedded, *, n_neighbors=5, metric='euclidean'):
    r"""Expresses to what extent the local structure is retained.
    The trustworthiness is within [0, 1]. It is defined as
    .. math::
        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))
    where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
    neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
    nearest neighbor in the input space. In other words, any unexpected nearest
    neighbors in the output space are penalised in proportion to their rank in
    the input space.
    * "Neighborhood Preservation in Nonlinear Projection Methods: An
      Experimental Study"
      J. Venna, S. Kaski
    * "Learning a Parametric Embedding by Preserving Local Structure"
      L.J.P. van der Maaten
    Parameters
    ----------
    X : array, shape (n_samples, n_features) or (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.
    X_embedded : array, shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.
    n_neighbors : int, optional (default: 5)
        Number of neighbors k that will be considered.
    metric : string, or callable, optional, default 'euclidean'
        Which metric to use for computing pairwise distances between samples
        from the original input space. If metric is 'precomputed', X must be a
        matrix of pairwise distances or squared distances. Otherwise, see the
        documentation of argument metric in sklearn.pairwise.pairwise_distances
        for a list of available metrics.
        .. versionadded:: 0.20
    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.
    """
    dist_X = pairwise_distances(X, metric=metric)
    if metric == 'precomputed':
        dist_X = dist_X.copy()
    # we set the diagonal to np.inf to exclude the points themselves from
    # their own neighborhood
    np.fill_diagonal(dist_X, np.inf)
    ind_X = np.argsort(dist_X, axis=1)
    # `ind_X[i]` is the index of sorted distances between i and other samples
    ind_X_embedded = NearestNeighbors(n_neighbors=n_neighbors).fit(
            X_embedded).kneighbors(return_distance=False)

    # We build an inverted index of neighbors in the input space: For sample i,
    # we define `inverted_index[i]` as the inverted index of sorted distances:
    # inverted_index[i][ind_X[i]] = np.arange(1, n_sample + 1)
    n_samples = X.shape[0]
    inverted_index = np.zeros((n_samples, n_samples), dtype=int)
    ordered_indices = np.arange(n_samples + 1)
    inverted_index[ordered_indices[:-1, np.newaxis],
                   ind_X] = ordered_indices[1:]
    ranks = inverted_index[ordered_indices[:-1, np.newaxis],
                           ind_X_embedded] - n_neighbors
    t = np.sum(ranks[ranks > 0])
    t = 1.0 - t * (2.0 / (n_samples * n_neighbors *
                          (2.0 * n_samples - 3.0 * n_neighbors - 1.0)))
    return t


class TSNE(BaseEstimator):
    r"""t-distributed Stochastic Neighbor Embedding.
    t-SNE [1] is a tool to visualize high-dimensional data. It converts
    similarities between data points to joint probabilities and tries
    to minimize the Kullback-Leibler divergence between the joint
    probabilities of the low-dimensional embedding and the
    high-dimensional data. t-SNE has a cost function that is not convex,
    i.e. with different initializations we can get different results.
    It is highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high. This will suppress some
    noise and speed up the computation of pairwise distances between
    samples. For more tips see Laurens van der Maaten's FAQ [2].
    Read more in the :ref:`User Guide <t_sne>`.
    Parameters
    ----------
    n_components : int, optional (default: 2)
        Dimension of the embedded space.
    perplexity : float, optional (default: 30)
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significanlty
        different results.
    early_exaggeration : float, optional (default: 12.0)
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.
    exaggeration : float, optional (default: 1.0)
        Same as early_exaggeration but applied to later stages of t-SNE.
        There are works showing that modifying this term can make t-SNE 
        look like UMAP and Laplacian Eigenmaps. See umap.UMAP for details.
    learning_rate : float, optional (default: 200.0)
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers. If the cost function gets stuck in a bad local
        minimum increasing the learning rate may help.
    n_iter : int, optional (default: 1000)
        Maximum number of iterations for the optimization. Should be at
        least 250.
    n_iter_without_progress : int, optional (default: 300)
        Maximum number of iterations without progress before we abort the
        optimization, used after 250 initial iterations with early
        exaggeration. Note that progress is only checked every 50 iterations so
        this value is rounded to the next multiple of 50.
        .. versionadded:: 0.17
           parameter *n_iter_without_progress* to control stopping criteria.
    min_grad_norm : float, optional (default: 1e-7)
        If the gradient norm is below this threshold, the optimization will
        be stopped.
    metric : string or callable, optional
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.
    init : string or numpy array, optional (default: "random")
        Initialization of embedding. Possible options are 'random', 'pca',
        and a numpy array of shape (n_samples, n_components).
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization.
    verbose : int, optional (default: 0)
        Verbosity level.
    random_state : int, RandomState instance, default=None
        Determines the random number generator. Pass an int for reproducible
        results across multiple function calls. Note that different
        initializations might result in different local minima of the cost
        function. See :term: `Glossary <random_state>`.
    method : string (default: 'barnes_hut')
        By default the gradient calculation algorithm uses Barnes-Hut
        approximation running in O(NlogN) time. method='exact'
        will run on the slower, but exact, algorithm in O(N^2) time. The
        exact algorithm should be used when nearest-neighbor errors need
        to be better than 3%. However, the exact method cannot scale to
        millions of examples.
        .. versionadded:: 0.17
           Approximate optimization *method* via the Barnes-Hut.
    angle : float (default: 0.5)
        Only used if method='barnes_hut'
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.
    attr: string (default: 'none'):
        Attributions to perform.  Options:
        * 'none' : does not compute any attributions (so runs usual tSNE).
        * 'grad_xy' : returns gradients w.r.t. tSNE components
        * 'grad_norm' : returns gradients w.r.t. norm of tSNE components
        * 'mean_grad_xy' : returns mean of gradients (over all steps)
        * 'mean_grad_norm' : returns mean of norm of gradients (over all steps)
        * 'pos_neg_force' : returns gradients w.r.t. positive and negative forces (experimental)
        * 'kl_obj': returns gradients w.r.t KL objective
        * 'kl_obj_mean': returns mean of gradients  w.r.t KL objective (over all steps)
    checkpoint_every: list (default: [])
        list of timesteps of gradients and embeddings to return
        assumes list is in ascending order!
        empty list = only return gradients and embeddings at the end
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search. This parameter
        has no impact when ``metric="precomputed"`` or
        (``metric="euclidean"`` and ``method="exact"``).
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        .. versionadded:: 0.22
    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.
    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.
    n_iter_ : int
        Number of iterations run.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.manifold import TSNE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> X_embedded = TSNE(n_components=2).fit_transform(X)
    >>> X_embedded.shape
    (4, 2)
    References
    ----------
    [1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
        Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.
    [2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
        https://lvdmaaten.github.io/tsne/
    [3] L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms.
        Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
        https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf
    """
    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_N_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    @_deprecate_positional_args
    def __init__(self, n_components=2, *, perplexity=30.0,
                 early_exaggeration=12.0, exaggeration=1.0, learning_rate=200.0, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5,
                 attr='none', checkpoint_every=[], n_jobs=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.exaggeration = exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs

        self.attr = attr

        assert isinstance(checkpoint_every, list)

        self.checkpoint_every = checkpoint_every

    def _fit(self, X, skip_num_points=0):
        """Private function to fit the model using X as training data."""

        if self.method not in ['barnes_hut', 'exact']:
            raise ValueError("'method' must be 'barnes_hut' or 'exact'")
        if self.angle < 0.0 or self.angle > 1.0:
            raise ValueError("'angle' must be between 0.0 - 1.0")
        if self.method == 'barnes_hut':
            X = self._validate_data(X, accept_sparse=['csr'],
                                    ensure_min_samples=2,
                                    dtype=[np.float32, np.float64])
        else:
            X = self._validate_data(X, accept_sparse=['csr', 'csc', 'coo'],
                                    dtype=[np.float32, np.float64])
        if self.metric == "precomputed":
            if self.attr != 'none':
                raise ValueError('Cannot use precomputed distance when computing attribution!')
            if isinstance(self.init, str) and self.init == 'pca':
                raise ValueError("The parameter init=\"pca\" cannot be "
                                 "used with metric=\"precomputed\".")
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")

            check_non_negative(X, "TSNE.fit(). With metric='precomputed', X "
                                  "should contain positive distances.")

            if self.method == "exact" and issparse(X):
                raise TypeError(
                    'TSNE with method="exact" does not accept sparse '
                    'precomputed distance matrix. Use method="barnes_hut" '
                    'or provide the dense distance matrix.')

        if self.method == 'barnes_hut' and self.n_components > 3:
            raise ValueError("'n_components' should be inferior to 4 for the "
                             "barnes_hut algorithm as it relies on "
                             "quad-tree or oct-tree.")
        random_state = check_random_state(self.random_state)

        if self.early_exaggeration < 1.0:
            raise ValueError("early_exaggeration must be at least 1, but is {}"
                             .format(self.early_exaggeration))

        if self.n_iter <= 250:
            raise ValueError("n_iter should be > 250")

        if self.attr not in ['none', 'grad_xy', 'grad_norm', 'mean_grad_xy', 'mean_grad_norm', 'pos_neg_force', 'kl_obj', 'kl_obj_mean']:
             raise ValueError("'attr' must be: 'none', 'grad_xy', 'grad_norm', 'mean_grad_xy', 'mean_grad_norm' or 'pos_neg_force', 'kl_obj', 'kl_obj_mean'")

        n_samples = X.shape[0]

        neighbors_nn = None

        if (self.attr != 'none') & (self.metric != "euclidean"):
            raise ValueError('Cannot use non-euclidean distance when computing attributions!')

        if self.method == "exact":
            # Retrieve the distance matrix, either using the precomputed one or
            # computing it.
            if self.metric == "precomputed":
                distances = X
            else:
                if self.verbose:
                    print("[t-SNE] Computing pairwise distances...")

                if self.metric == "euclidean":
                    distances = pairwise_distances(X, metric=self.metric,
                                                   squared=True)
                else:
                    distances = pairwise_distances(X, metric=self.metric,
                                                   n_jobs=self.n_jobs)

                if np.any(distances < 0):
                    raise ValueError("All distances should be positive, the "
                                     "metric given is not correct")

            # compute the joint probability distribution for the input space
            P, conditional_P, betas = _joint_probabilities(distances, self.perplexity, self.verbose)
            assert np.all(np.isfinite(P)), "All probabilities should be finite"
            assert np.all(P >= 0), "All probabilities should be non-negative"
            assert np.all(P <= 1), ("All probabilities should be less "
                                    "or then equal to one")

            if self.attr != 'none':
                _, dP = _compute_dp(X, conditional_P, P, betas)
            else:
                dP = None

        else:
            # Compute the number of nearest neighbors to find.
            # LvdM uses 3 * perplexity as the number of neighbors.
            # In the event that we have very small # of points
            # set the neighbors to n - 1.
            n_neighbors = min(n_samples - 1, int(3. * self.perplexity + 1))

            if self.verbose:
                print("[t-SNE] Computing {} nearest neighbors..."
                      .format(n_neighbors))

            # Find the nearest neighbors for every point
            knn = NearestNeighbors(algorithm='auto',
                                   n_jobs=self.n_jobs,
                                   n_neighbors=n_neighbors,
                                   metric=self.metric)
            t0 = time()
            knn.fit(X)
            duration = time() - t0
            if self.verbose:
                print("[t-SNE] Indexed {} samples in {:.3f}s...".format(
                    n_samples, duration))

            t0 = time()
            distances_nn = knn.kneighbors_graph(mode='distance')
            duration = time() - t0
            if self.verbose:
                print("[t-SNE] Computed neighbors for {} samples "
                      "in {:.3f}s...".format(n_samples, duration))

            # Free the memory used by the ball_tree
            del knn

            if self.metric == "euclidean":
                # knn return the euclidean distance but we need it squared
                # to be consistent with the 'exact' method. Note that the
                # the method was derived using the euclidean method as in the
                # input space. Not sure of the implication of using a different
                # metric.
                distances_nn.data **= 2

            # compute the joint probability distribution for the input space
            P, conditional_P, betas = _joint_probabilities_nn(distances_nn,
                                                              self.perplexity,
                                                              self.verbose)

            # compute dP
            if self.attr != 'none':
                t0 = time()
                _, dP = _compute_dp_bh(X,
                                       conditional_P,
                                       P,
                                       betas,
                                       max(self.n_components - 1, 1),  # degrees of freedom (see below for justification)
                                       _openmp_effective_n_threads())
                duration = time() - t0
                if self.verbose:
                    print("[t-SNE] Computed gradient of P for {} samples "
                          "in {:.3f}s...".format(n_samples, duration))
            else:
                dP = None

        if self.attr != 'none':
            self.attr_shape = [self.n_components, n_samples, dP.shape[-1]]
        if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == 'pca':
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=random_state)
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)

            # PCA is rescaled so that PC1 has standard deviation 1e-4 which is
            # the default value for random initialization. See issue #18018.
            X_embedded = X_embedded / np.std(X_embedded[:, 0]) * 1e-4

            # store this for attribution calculations
            # Note that this will be the identity matrix when input is PCA transformed
            # It will be the unscaled loadings matrix otherwise
            if self.attr != 'none':
                self.attr_init = np.stack([pca.components_]*n_samples, 1)
        elif self.init == 'random':
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = 1e-4 * random_state.randn(
                n_samples, self.n_components).astype(np.float32)
            if self.attr != 'none':
                self.attr_init = np.zeros(shape=self.attr_shape, dtype=np.float32)
        else:
            raise ValueError("'init' must be 'pca', 'random', or "
                             "a numpy array")

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1, 1)

        # Not sure how degrees_of_freedom!=1 affects gradient computations!
        assert degrees_of_freedom == 1; 'Gradient computations only valid for degrees of freedom = 1'

        return self._tsne(P, degrees_of_freedom, n_samples,
                          X_embedded=X_embedded,
                          neighbors=neighbors_nn,
                          skip_num_points=skip_num_points,
                          dP=dP)

    def _tsne(self, P, degrees_of_freedom, n_samples, X_embedded,
              neighbors=None, skip_num_points=0, dP=None):
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with two stages:
        # * initial optimization with early exaggeration and momentum at 0.5
        # * final optimization with momentum at 0.8

        params = X_embedded.ravel()

        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points),
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_N_ITER,
            "n_iter": self._EXPLORATION_N_ITER,
            "momentum": 0.5,
        }
        if self.method == 'barnes_hut':
            obj_func = _kl_divergence_bh
            opt_args['kwargs']['angle'] = self.angle
            # Repeat verbose argument for _kl_divergence_bh
            opt_args['kwargs']['verbose'] = self.verbose
            # Get the number of threads for gradient computation here to
            # avoid recomputing it at each iteration.
            opt_args['kwargs']['num_threads'] = _openmp_effective_n_threads()
            attr_func = _compute_attr_bh

        else:
            obj_func = _kl_divergence
            attr_func = _compute_attr

        # store what is needed for gradient calculations
        attr_args = {}
        attr_args['verbose'] = self.verbose
        attr_args['num_threads'] = _openmp_effective_n_threads()
        attr_args['angle'] = self.angle
        opt_args['attr_args'] = attr_args

        # store intermediate values for attribution computations!
        if self.attr != 'none':
            dYx = np.array(self.attr_init, dtype=np.float32)  # (n_components, n_samples, n_feats)
            diYx = np.zeros(shape=self.attr_shape, dtype=np.float32)
            #attr_args['dP'] = dP
        else:
            dYx, diYx = None, None

        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exaggeration parameter
        P *= self.early_exaggeration
        if dP is not None:
            dP *= self.early_exaggeration

        # Stores intermediate gradients and attributions
        return_obj = {}
        return_obj['attrs'] = []
        return_obj['embeddings'] = []
        params, kl_divergence, it, dYx, diYx, return_obj = _gradient_descent(obj_func,
                                                                             attr_func,
                                                                             params,
                                                                             dP,
                                                                             dYx,
                                                                             diYx,
                                                                             self.attr,
                                                                             self.checkpoint_every,
                                                                             return_obj,
                                                                             **opt_args)

        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations with early "
                  "exaggeration: %f" % (it + 1, kl_divergence))

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P /= self.early_exaggeration
        if dP is not None:
            dP /= self.early_exaggeration

        # Now multiply by exaggeration
        P *= self.exaggeration
        if dP is not None:
            dP *= self.exaggeration

        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if it < self._EXPLORATION_N_ITER or remaining > 0:
            opt_args['n_iter'] = self.n_iter
            opt_args['it'] = it + 1
            opt_args['momentum'] = 0.8
            opt_args['n_iter_without_progress'] = self.n_iter_without_progress
            params, kl_divergence, it, dYx, diYx, return_obj = _gradient_descent(obj_func,
                                                                                 attr_func,
                                                                                 params,
                                                                                 dP,
                                                                                 dYx,
                                                                                 diYx,
                                                                                 self.attr,
                                                                                 self.checkpoint_every,
                                                                                 return_obj,
                                                                                 **opt_args)

        # Save the final number of iterations
        self.n_iter_ = it

        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations: %f"
                  % (it + 1, kl_divergence))

        #X_embedded = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence

        return return_obj

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed
        output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.
        y : Ignored
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        embedding = self._fit(X)
        self.embedding_ = embedding
        return self.embedding_

    def fit(self, X, y=None):
        """Fit X into an embedded space.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.
        y : Ignored
        """
        self.fit_transform(X)
        return self
