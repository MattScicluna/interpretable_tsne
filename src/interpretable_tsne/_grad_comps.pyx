# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
# This is a modification of the tSNE implementation from sklearn:
# https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/manifold/_barnes_hut_tsne.pyx
#
# Author: Matthew Scicluna -- <mattcscicluna@gmail.com>
# Licence: BSD 3 clause (C) 2021

import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from libc.math cimport sqrt, log
from libc.stdlib cimport abort, malloc, free
from cython.parallel cimport prange, parallel

#from sklearn.neighbors._quad_tree cimport _QuadTree
from ._quad_tree cimport _QuadTree

np.import_array()


cdef char* EMPTY_STRING = ""

cdef extern from "math.h":
    float fabsf(float x) nogil

# Smallest strictly positive value that can be represented by floating
# point numbers for different precision levels. This is useful to avoid
# taking the log of zero when computing the KL divergence.
cdef float FLOAT32_TINY = np.finfo(np.float32).tiny

# Useful to void division by zero or divergence to +inf.
cdef float FLOAT64_EPS = np.finfo(np.float64).eps

# This is effectively an ifdef statement in Cython
# It allows us to write printf debugging lines
# and remove them at compile time
cdef enum:
    DEBUGFLAG = 0

cdef extern from "time.h":
    # Declare only what is necessary from `tm` structure.
    ctypedef long clock_t
    clock_t clock() nogil
    double CLOCKS_PER_SEC


def compute_p_gradient(double [:, :] X,
                       long [:] neigh_P_ij, # (1368994,)
                       long [:] indptr_P_ij,  # (11315,)
                       float [:] P_ij, # P_{i,j} of size (1368994,)
                       long [:] neigh_P_ji, # these are indices for P_{j|i} (1368994,)
                       long [:] indptr_P_ji, # (11315,)
                       float [:] P_ji, # P_{j|i} of size (1368994,)
                       long [:] neigh_P, # (2252376,)
                       long [:] indptr_P, # (11315,)
                       double [:] betas,
                       int num_threads):
    """
    Computes dP/dx by first computing EX and then running `_compute_p_gradient`
    """

    cdef:
        int n_dimensions = X.shape[1]
        int n_samples = X.shape[0]
        int n_nonzero = len(neigh_P)
        np.ndarray[np.float64_t, ndim=2] EX = np.zeros((n_samples, n_dimensions), dtype=np.float64)
        np.ndarray[np.float64_t, ndim=2] dP = np.zeros((n_nonzero, n_dimensions), dtype=np.float64)
    
    # intermediate term needed to compute dP/dx
    _compute_ex(X, EX, neigh_P_ji, indptr_P_ji, P_ji, betas, n_dimensions, num_threads)

    # now we can compute dP/dx
    _compute_p_gradient(X,
                        EX,
                        dP,
                        neigh_P_ij,
                        indptr_P_ij,
                        P_ij,
                        neigh_P_ji,
                        indptr_P_ji,
                        P_ji,
                        neigh_P,
                        indptr_P,
                        betas,
                        n_dimensions,
                        num_threads)

    return EX, dP

cdef void _compute_ex(double [:, :] X,
                      double [:, :] EX,
                      long [:] neigh_P_ji, # these are indices for P_{j|i}
                      long [:] indptr_P_ji, # these are indices where next column starts
                      float [:] P_ji, # P_{i|j}
                      double [:] betas,
                      int n_dimensions,
                      int num_threads) nogil:

    cdef:
        long i, j, k, m
        float pji, dijk
        int n_samples = len(X) #X.shape[0] #

    # compute EX
    with nogil, parallel(num_threads=num_threads):
        for i in prange(0, n_samples, schedule='static'):
            for k in range(indptr_P_ji[i], indptr_P_ji[i+1]):
                j = neigh_P_ji[k]
                pji = P_ji[k]
                for m in range(n_dimensions):
                    dijk = -(X[i, m]-X[j, m])*betas[i]
                    EX[i, m] += pji*dijk

                    
cdef void _compute_p_gradient(double [:, :] X,
                              double [:, :] EX,
                              double [:, :] dP,
                              long [:] neigh_cond_P, # (1368994,)
                              long [:] indptr_cond_P,  # (11315,)
                              float [:] cond_P, # P_{i,j} of size (1368994,)
                              long [:] neigh_cond_P_T, # these are indices for P_{j|i} (1368994,)
                              long [:] indptr_cond_P_T, # (11315,)
                              float [:] cond_P_T, # P_{j,i} of size (1368994,)
                              long [:] neigh_P, # (2252376,)
                              long [:] indptr_P, # (11315,)
                              double [:] betas,
                              int n_dimensions,
                              int num_threads):

    # Sum over the following expression for i not equal to j
    # grad P_{ij} = p_{j|i}(d_{ij}-\sum_{k \ne i} p_{k|i} d_{i|k} ) + d_{ji}p_{i|j}(1-p_{i|j})
    cdef:
        long i, k, m, j, j2, j3, k2, k3
        long idx_icp, idx_icp_T
        float pji, pij, dijm, djim
        int n_samples = len(X)
        int n_nonzero = len(neigh_cond_P)

    # compute dP
    for i in range(0, n_samples):
        idx_icp = 0
        idx_icp_T = 0
        icp = list(range(indptr_cond_P[i], indptr_cond_P[i+1])) # get neighbours for cond_P
        icpt = list(range(indptr_cond_P_T[i], indptr_cond_P_T[i+1])) # get neighbours for cond_P_T
        for k in range(indptr_P[i], indptr_P[i+1]):
            if len(icp) > idx_icp:
                k2 = icp[idx_icp]
            if len(icpt) > idx_icp_T:
                k3 = icpt[idx_icp_T]
            j = neigh_P[k]
            j2 = neigh_cond_P[k2]
            j3 = neigh_cond_P_T[k3]

            if j == j2:
                pij = cond_P[k2]
                idx_icp += 1
                for m in range(n_dimensions):
                    djim = -(X[j, m]-X[i, m])*betas[j]
                    dP[k, m] += djim*pij*(1-pij)
            if j == j3:
                pji = cond_P_T[k3]
                idx_icp_T += 1
                for m in range(n_dimensions):
                    dijm = -(X[i, m]-X[j, m])*betas[i]
                    dP[k, m] += pji*(dijm-EX[i, m])
            for m in range(n_dimensions):
                dP[k, m] /= (2*n_samples)


cdef void _compute_pos_attr(float [:, :] Y,
                            float [:, :, :] dYx,
                            float* pos_f,
                            float [:, :] dP,
                            long [:] dP_idxs, 
                            long [:] dP_idptr,
                            float [:] P,
                            int dof,
                            int num_threads) nogil:
    """
    Computes 'positive gradient'
    This is the portion of the gradient which 
    exploits the sparsity of P (and dP)
    """

    cdef:
        long i, k, j, c, d, l
        float p_ij, dp_ij, dij, qijZ
        float float_dof = (float) (dof)
        long n_samples = Y.shape[0]
        long n_components = dYx.shape[0]
        long n_dimensions = dYx.shape[2]

    with nogil, parallel(num_threads=num_threads):
        ydiff = <float *> malloc(sizeof(float) * n_components)
        phi = <float *> malloc(sizeof(float) * n_components)
        phi_grad = <float *> malloc(sizeof(float) * n_components * n_components)

        for i in prange(0, n_samples, schedule='static'):
            #  Initialize gradient
            for c in range(n_components):
                for d in range(n_dimensions):
                    pos_f[c + n_components * (i + n_samples * d)] = 0.0

            for k in range(dP_idptr[i], dP_idptr[i+1]):
                #  Get neighbour
                j = dP_idxs[k]
                p_ij = P[k]

                # compute phi_ijc
                dij = 0.0
                for c in range(n_components):
                    ydiff[c] = Y[i, c] - Y[j, c]
                    dij += ydiff[c] * ydiff[c]

                qijZ = float_dof / (float_dof + dij)
                dij = qijZ

                # fill in pos_f using phi_ijc, dphi_ijc,  p_ij and dP
                for c in range(n_components):
                    phi[c] = ydiff[c]*dij

                # compute dphi_ijc
                for c in range(n_components):
                    for l in range(n_components):
                        phi_grad[c*n_components+l] = -2*phi[c]*phi[l]
                        if c == l:
                            phi_grad[c*n_components+l] += dij

                # fill in pos_f using phi_ijc, dphi_ijc,  p_ij and dP
                for c in range(n_components):
                    for d in range(n_dimensions):
                        # dp_{ij}/dx_{id} * phi_{ijc}
                        pos_f[c + n_components * (i + n_samples * d)] += dP[k, d]*phi[c]
                        # p_ij * dphi_{ijc}/dx_{id} = p_ij * \sum_{l=0}^{1} dphi_{ijc}/dy_{il} * dy_{il}/dx_{id}
                        for l in range(n_components):
                            pos_f[c + n_components * (i + n_samples * d)] += p_ij*phi_grad[c*n_components+l]*dYx[l, i, d]

        free(ydiff)
        free(phi)
        free(phi_grad)


cdef void _compute_eq(float [:, :] Y,
                      float [:, :] E_qphi,
                      float [:, :] q,
                      float [:, :, :] phi,
                      float [:, :] sizes,
                      long [:] idxs,
                      _QuadTree qt,
                      float Z,
                      int dof,
                      long n_samples,
                      long n_components,
                      float theta,
                      int num_threads) nogil:
    """
    Compute E_q_{cells}[phi_{i,cell,c}] (also stores q_{i,cell}, N_{cell} and phi_{i,cell,c})
    """

    cdef:
        long c, j, idx, i
        float float_dof = (float) (dof)
        float exponent = (dof + 1.0) / 2.0
        float size, dist2s, qijZ
        float q_ij, phi_ijc
        int offset = n_components + 2

    with nogil, parallel(num_threads=num_threads):
        # Define thread-local buffers
        pos = <float *> malloc(sizeof(float) * n_components)
        summary = <float*> malloc(sizeof(float) * n_samples * offset)

        for i in prange(0, n_samples, schedule='static'):
            for c in range(n_components):
                pos[c] = Y[i, c]

            idx = qt.summarize(pos, summary, theta*theta)
            idxs[i] = idx
            for j in range(idx // offset):
                dist2s = summary[j * offset + n_components] # n_dimensions
                size = summary[j * offset + n_components + 1] # n_dimensions
                qijZ = float_dof / (float_dof + dist2s)  # 1/(1+dist)

                if dof != 1:  # i.e. exponent != 1
                    qijZ = qijZ ** exponent

                q_ij = qijZ / Z

                #  Store this computation for the next parts!
                q[i, j] = q_ij
                sizes[i, j] = size

                for c in range(n_components):
                    phi_ijc = summary[j * offset + c]*qijZ
                    phi[c, i, j] = phi_ijc  # also store this!
                    E_qphi[i, c] += size * phi_ijc * q_ij


cdef void __compute_neg_attr(float [:, :] Y,
                            float [:, :, :] dYx,
                            float* neg_f,
                            float [:, :] E_qphi, # E_q_{cells}[phi_{i,cell,c}]
                            float [:, :] q, # q_{i,cell}
                            float [:, :, :] phi, # phi_{i,cell,c})
                            float [:, :] sizes, # N_{cell} (number of points in cell)
                            long [:] idxs, # number of cells
                            float Z,
                            int num_threads) nogil:

    cdef:
        long c, i, d, j, l, idx
        long n_samples = Y.shape[0]
        long n_components = dYx.shape[0] # qt.n_dimensions (this is called n_dimensions elsewhere!)
        long n_dimensions = dYx.shape[2]
        int offset = n_components + 2
        float dq_phi_N, dphi_q_N

    with nogil, parallel(num_threads=num_threads):
        # Define thread-local buffers
        dq_ijy_i = <float *> malloc(sizeof(float) * n_components)
        dphi_ijy_i = <float *> malloc(sizeof(float) * n_components * n_components)

        for i in prange(0, n_samples, schedule='static'):
            #  Initialize gradient
            for c in range(n_components):
                for d in range(n_dimensions):
                    neg_f[c + n_components * (i + n_samples * d)] = 0.0

            # Clear the arrays
            for c in range(n_components):
                E_qphi[c] = 0.0
                dq_ijy_i[c] = 0.0
            
            idx = idxs[i]

            for j in range(idx // offset):
                for l in range(n_components):
                    dq_ijy_i[l] = -2*q[i, j]
                    dq_ijy_i[l] *= (phi[l, i , j] - 2*E_qphi[i, l])

                #  compute dq_{i,cell}/dx_{i,d} * phi_{i,cell,c}
                for c in range(n_components):
                    for l in range(n_components):
                        dq_phi_N = dq_ijy_i[l]*phi[c, i, j]*sizes[i, j]  # multiply by phi_{i,cell,c} and N_{cell} here!
                        for d in range(n_dimensions):
                            # dq_{i,cell}/dx_{i,d} = \sum_{l=0}^{1} dq_{i,cell}/dy_{i,l} * dy_{i,l}/dx_{i,d}
                            neg_f[c + n_components * (i + n_samples * d)] += dq_phi_N*dYx[l, i, d]

                #  compute dphi_{i,cell,c}/dx_{i,d} * q_{i,cell}
                for c in range(n_components):
                    # compute dphi_{i,cell,c}/dy_{i,l}
                    for l in range(n_components):
                        dphi_ijy_i[c*n_components+l] = -2*phi[c, i, j]*phi[l, i, j]
                        if c == l:
                            dphi_ijy_i[c*n_components+l] += q[i, j]*Z
                    # compute dphi_{i,cell,c}/dy_{i,l} * dy_{i,l}/dx_{i,d}
                    for l in range(n_components):
                        dphi_q_N = dphi_ijy_i[c*n_components+l]*q[i, j]*sizes[i, j]  # multiply by q_{i, cell} and N_{cell} here! 
                        for d in range(n_dimensions):
                            neg_f[c + n_components * (i + n_samples * d)] += dphi_q_N*dYx[l, i, d]

        free(dq_ijy_i)
        free(dphi_ijy_i)


cdef void _compute_neg_attr(float [:, :] Y,
                            float [:, :, :] dYx,
                            float* neg_f,
                            _QuadTree qt,
                            float Z,
                            int dof,
                            float theta,
                            int num_threads):
    """
    Computes 'negative gradient'
    This is the portion of the gradient which 
    uses the quadtree
    """

    cdef:
        long n_samples = Y.shape[0]
        long n_components = dYx.shape[0]
        long n_dimensions = dYx.shape[2]
        np.ndarray[np.float32_t, ndim=2] E_qphi = np.zeros((n_samples, n_components), dtype=np.float32)
        np.ndarray[np.float32_t, ndim=2] q = np.zeros((n_samples, n_samples), dtype=np.float32)
        np.ndarray[np.float32_t, ndim=3] phi = np.zeros((n_components, n_samples, n_samples), dtype=np.float32)
        np.ndarray[np.float32_t, ndim=2] sizes = np.zeros((n_samples, n_samples), dtype=np.float32)
        np.ndarray[np.int64_t, ndim=1] idxs = np.zeros((n_samples), dtype=np.int64)

    _compute_eq(Y, E_qphi, q, phi, sizes, idxs, qt, Z, dof, n_samples, n_components, theta, num_threads)
    __compute_neg_attr(Y, dYx, neg_f, E_qphi, q, phi, sizes, idxs, Z, num_threads)


def compute_attr(float [:, :] Y, # (11314,2)
                 _QuadTree qt,
                 float sumQ,
                 float [:, :, :] dYx, # (2, 11314, 50)
                 float [:, :] dP, # (2252376, 50)
                 long [:] dP_idxs, # (2252376,)
                 long [:] dP_idptr, # (11315,)
                 float [:] P, # (2252376,)
                 float angle,
                 int dof,
                 int verbose,
                 int num_threads):

    cdef:
        long i, j, k, coord
        int n_dimensions = dYx.shape[2]
        int n_samples = Y.shape[0]
        int n_nonzero = len(P)
        int n_components = Y.shape[1]
        np.ndarray[np.float32_t, ndim=3] new_dYx = np.zeros((n_components, n_samples, n_dimensions), dtype=np.float32)

        # Add for diagnostics
        np.ndarray[np.float32_t, ndim=3] pos_array = np.zeros((n_components, n_samples, n_dimensions), dtype=np.float32)
        np.ndarray[np.float32_t, ndim=3] neg_array = np.zeros((n_components, n_samples, n_dimensions), dtype=np.float32)

    cdef float* pos_f = <float*> malloc(sizeof(float) * n_samples * n_dimensions * n_components)
    cdef float* neg_f = <float*> malloc(sizeof(float) * n_samples * n_dimensions * n_components)

    _compute_pos_attr(Y, dYx, pos_f, dP, dP_idxs, dP_idptr, P, dof, num_threads)
    _compute_neg_attr(Y, dYx, neg_f, qt, sumQ, dof, angle, num_threads)

    for i in prange(0, n_samples, nogil=True, num_threads=num_threads, schedule='static'):
        for j in range(n_components):
            for k in range(n_dimensions):
                coord = j + n_components * (i + n_samples * k)
                new_dYx[j, i, k] = pos_f[coord] - neg_f[coord]
                pos_array[j, i, k] = pos_f[coord]
                neg_array[j, i, k] = neg_f[coord]

    free(pos_f)
    free(neg_f)

    return new_dYx, pos_array, neg_array


cpdef _compute_q_phi_debug(float [:, :] Y,
                            _QuadTree qt,
                            float theta,
                            float Z,
                            int dof):
    """
    This just computes the matrices needed to compute 
    the negative gradient using matrix multiplies
    (we use these in our unittests)
    """

    #return qt._get_cell_ndarray()
    cdef:
        long i, j, c
        float float_dof = (float) (dof)
        float exponent = (dof + 1.0) / 2.0
        float size, dist2s, qijZ, q_ij
        long n_samples = Y.shape[0]
        long n_components = Y.shape[1] # qt.n_dimensions (this is called n_dimensions elsewhere!)
        int offset = n_components + 2
        
        # Add for diagnostics
        np.ndarray[np.float32_t, ndim=2] Q = np.zeros((n_samples, n_samples), dtype=np.float32)
        np.ndarray[np.float32_t, ndim=2] D = np.zeros((n_samples, n_samples), dtype=np.float32)
        np.ndarray[np.float32_t, ndim=3] Ydiff = np.zeros((n_components, n_samples, n_samples), dtype=np.float32)
        np.ndarray[np.float32_t, ndim=2] S = np.zeros((n_samples, n_samples), dtype=np.float32)

    summary = <float*> malloc(sizeof(float) * n_samples * offset)
    pos = <float *> malloc(sizeof(float) * n_components)

    for i in range(0, n_samples):

        for c in range(n_components):
            pos[c] = Y[i, c]

        # Find which nodes are summarizing and collect their centers of mass
        # deltas, and sizes, into vectorized arrays
        idx = qt.summarize(pos, summary, theta*theta)

        # Compute E_q_{cells}[phi_{i,cell,c}] (also stores q_{i,cell}, N_{cell} and phi_{i,cell,c})
        for j in range(idx // offset):

            dist2s = summary[j * offset + n_components]
            size = summary[j * offset + n_components + 1]
            qijZ = float_dof / (float_dof + dist2s)  # 1/(1+dist)

            if dof != 1:  # i.e. exponent != 1
                qijZ = qijZ ** exponent

            q_ij = qijZ / Z

            #  Store these computations for unittests!
            D[i, j] = dist2s
            Q[i, j] = q_ij
            S[i, j] = size

            for c in range(n_components):
                Ydiff[c, i, j] = summary[j * offset + c]

    return Q, D, Ydiff, S
