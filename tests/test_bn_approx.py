'''
This unittest tests the Barnes-Hut approximation of the t-SNE gradient computations
'''

import unittest
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csr_matrix
import numpy as np

#from _grad_comps import *
from src.interpretable_tsne.tsne import pairwise_distances, _joint_probabilities, _compute_dp, _openmp_effective_n_threads, _joint_probabilities_nn, _compute_dp_bh, NearestNeighbors


class Test_compute_dp_bh(unittest.TestCase):

    def setUp(self):

        # for reproducibility
        np.random.seed(seed=42)

        perplexity = 30
        X = np.random.randn(1000, 50)
        n_neighbors = min(1000 - 1, int(3. * perplexity + 1))

        # Find the nearest neighbors for every point
        knn = NearestNeighbors(algorithm='auto',
                               n_jobs=8,
                               n_neighbors=n_neighbors,
                               metric='euclidean')
        knn.fit(X)
        distances_nn = knn.kneighbors_graph(mode='distance')
        del knn
        distances_nn.data **= 2
        self.distances_nn = distances_nn

        P, P_ji, betas = _joint_probabilities_nn(distances_nn, perplexity, True)

        EX, dP = _compute_dp_bh(X,
                                P_ji,
                                P,
                                betas,
                                1,  # degrees of freedom (see below for justification)
                                _openmp_effective_n_threads())

        self.EX = EX
        self.dP = dP
        self.X = X
        self.betas = betas
        self.P = P
        self.P_ji = P_ji

    def test_dP(self):
        
        data, perplexity=30
        n_samples = data.shape[0]
        distances = pairwise_distances(data, metric="euclidean", squared=True)
        P, conditional_P, betas = _joint_probabilities(distances, perplexity, verbose=2)
        _, dP = _compute_dp(data, conditional_P, P, betas)

        n_neighbors = min(n_samples - 1, int(3. * perplexity + 1))

        # Find the nearest neighbors for every point
        knn = NearestNeighbors(algorithm='auto',
                               n_jobs=8,
                               n_neighbors=n_neighbors,
                               metric="euclidean")
        knn.fit(data)
        distances_nn = knn.kneighbors_graph(mode='distance')

        # Free the memory used by the ball_tree
        del knn

        distances_nn.data **= 2

        # compute the joint probability distribution for the input space
        P2, conditional_P2, betas2 = _joint_probabilities_nn(distances_nn,
                                                          perplexity,
                                                          2)

        # compute dP
        _, dP2 = _compute_dp_bh(data,
                               conditional_P2,
                               P2,
                               betas2,
                               max(2 - 1, 1),  # degrees of freedom (see below for
                               _openmp_effective_n_threads())


        dense_mats = [csr_matrix((dP2[:,i], P2.indices, P2.indptr), shape=(n_samples, n_samples)).toarray() for i in range(data.shape[1])]
        dP3 = np.stack(dense_mats)
        dP3 = dP3.transpose(1, 2, 0)
        #dP3 = dP3.reshape(n_samples, n_samples, data.shape[1])
        return dP, dP3, P, P2

    def test_dP(self):

        # compute dP the long way
        _P_ji = np.expand_dims(self.P_ji.todense(), 2)
        _P_ij = np.expand_dims(self.P_ji.todense().T, 2)

        dij = -((np.expand_dims(self.X, 1)-np.expand_dims(self.X, 0)))*np.expand_dims(np.expand_dims(self.betas, 1), 1)

        # ensure that dij[i,j,d] = -(x_id - x_jd) * beta_i
        j = self.P_ji.indices[0]
        self.assertTrue((dij[0, j] == -(self.X[0]-self.X[j])*self.betas[0]).all(),
                       '`dij` matrix does not match hand calculation at i=0, j={}'.format(j))

        # Verify EX
        self.assertTrue(np.isclose((_P_ji*dij).sum(1), self.EX).mean() == 1.0,
                       'EX term from `_compute_dp_bh` does not match hand calculation')

        # Now we verify dP
        dP_ji = _P_ji*(dij - np.expand_dims((dij*_P_ji).sum(1), 1))
        dji = ((np.expand_dims(self.X, 1)-np.expand_dims(self.X, 0)))*np.expand_dims(np.expand_dims(self.betas, 1), 0)

        dP_ij = dji*_P_ij*(1-_P_ij)

        dP = (dP_ij + dP_ji)/(2*1000)

        # test first 10 rows for equality
        k = 0
        for i in range(0, 10):
            for j in range(self.P.indptr[i], self.P.indptr[i+1]):
                l = self.P.indices[j]
                self.assertTrue(np.isclose(self.dP[k], dP[i, l]).all(),
                               'dP term from `_compute_dp_bh` does not match hand calculation for i={}, j={}'.format(i, l))
                k += 1


class Test_compute_attr_bh(unittest.TestCase):

    def setUp(self):
        # for reproducibility
        np.random.seed(seed=42)

        perplexity = 30
        X = np.random.randn(1000, 50)
        n_neighbors = min(1000 - 1, int(3. * perplexity + 1))

        # Find the nearest neighbors for every point
        knn = NearestNeighbors(algorithm='auto',
                               n_jobs=8,
                               n_neighbors=n_neighbors,
                               metric='euclidean')
        knn.fit(X)
        distances_nn = knn.kneighbors_graph(mode='distance')
        del knn
        distances_nn.data **= 2

        P, P_ji, betas = _joint_probabilities_nn(distances_nn, perplexity, True)

        _, dP = _compute_dp_bh(X,
                               P_ji,
                               P,
                               betas,
                               1,  # degrees of freedom (see below for justification)
                               _openmp_effective_n_threads())

        self.P = P
        self.dP = dP

        # make dP dense
        dP2 = [np.array(csr_matrix((self.dP[:,i],
                                    self.P.indices,
                                    self.P.indptr),
                                   shape=(1000,1000)).todense())
               for i in range(50)]
        dP2 = np.stack(dP2, 2)
        self.dP2 = dP2

    def test_attr_first_step(self):

        Y = np.random.randn(1000, 2).astype(np.float32) * 1e-4
        dYx = np.zeros(shape=(2, 1000, 50)).astype(np.float32)

        error, grad, qt, sumQ = _kl_divergence_bh(Y,
                                                  self.P,
                                                  1,
                                                  1000,
                                                  2,
                                                  angle=0.5,
                                                  skip_num_points=0,
                                                  verbose=False,
                                                  compute_error=True,
                                                  num_threads=_openmp_effective_n_threads())

        # Compute dQ, dPhi, Phi
        ddYx, pos_array, neg_array = _compute_attr_bh(Y,
                                                      qt,
                                                      sumQ,
                                                      dYx,
                                                      self.dP,
                                                      self.P,
                                                      1,
                                                      1000,
                                                      2,
                                                      0.5,
                                                      1,
                                                      _openmp_effective_n_threads())

        #mask = np.array(self.P.todense() != 0)  # need to ignore entries where P = 0
        phi = (np.expand_dims(Y, 1) - np.expand_dims(Y, 0)) / np.expand_dims(1 + pairwise_distances(Y)**2, 2)

        # (2, 1000, 1000, 50) -> (2, 1000, 50)
        ddYx_longform = (np.expand_dims(phi.transpose(2,0,1), 3) * np.expand_dims(self.dP2, 0)).sum(2)

        self.assertTrue(np.isclose(ddYx, ddYx_longform).all(),
                        'attribution from `_compute_attr_bh` does not match hand calculation in first step')

    def test_attr_second_step(self):

        Y = np.random.randn(1000, 2).astype(np.float32) * 1e-4
        dYx = np.random.randn(2, 1000, 50).astype(np.float32)  # previous step is no longer zero

        error, grad, qt, sumQ = _kl_divergence_bh(Y,
                                                  self.P,
                                                  1,
                                                  1000,
                                                  2,
                                                  angle=0.5,
                                                  skip_num_points=0,
                                                  verbose=False,
                                                  compute_error=True,
                                                  num_threads=_openmp_effective_n_threads())

        # Compute dQ, dPhi, Phi
        ddYx, pos_array, neg_array = _compute_attr_bh(Y,
                                                      qt,
                                                      sumQ,
                                                      dYx,
                                                      self.dP,
                                                      self.P,
                                                      1,
                                                      1000,
                                                      2,
                                                      0.5,
                                                      1,
                                                      _openmp_effective_n_threads())

        # computes Phi
        #mask = np.array(self.P.todense() != 0)  # need to ignore entries where P = 0
        D = 1 / (1 + pairwise_distances(Y)**2)
        phi = (np.expand_dims(Y, 1) - np.expand_dims(Y, 0)) * np.expand_dims(D, 2)  # (1000, 1000, 2)

        # computes dP/dx * phi
        # (2, 1000, 1000, 50) -> (2, 1000, 50)
        ddYx_longform = (np.expand_dims(phi.transpose(2,0,1), 3) * np.expand_dims(self.dP2, 0)).sum(2)

        # computes dPhi/dy
        dPhiy = (np.expand_dims(D, axis=(2,3)) * np.expand_dims(np.eye(phi.shape[2], phi.shape[2]), axis=(0,1)))-2*np.expand_dims(phi, 2)*np.expand_dims(phi, 3)

        # computes dPhi/dx
        # (1000, 1000, 1, 2, 2) @ (1000, 1, 50, 1, 2) -> (1000, 1000, 50, 2)
        dPhix = (np.expand_dims(dPhiy, 2) @ np.expand_dims(dYx.transpose(1,2,0), axis=(1,4))).squeeze()
        #(num.unsqueeze(2).unsqueeze(2)*(torch.eye(phi.shape[2]).unsqueeze(0).unsqueeze(0).to(device))-2*(phi.unsqueeze(2)*phi.unsqueeze(3)))
        #DPhix = (DPhiy.unsqueeze(2) @ prev_DYx.permute(1,2,0).unsqueeze(1).unsqueeze(4)).squeeze()

        # computes (dP/dx * phi + dPhi/dx * P) [this is the positive half of the gradient]
        # (2, 1000, 1000, 50) x (1, 1000, 1000, 1) -> (2, 1000, 50)
        ddYx_longform += (dPhix.transpose(3,0,1,2) * np.expand_dims(self.P.todense(), axis=(0,3))).sum(2)

        self.assertTrue(np.isclose(pos_array, ddYx_longform).all(),
                       'attribution from `_compute_attr_bh` does not match hand calculation in general')

        # computes negative attr
        #from _grad_comps import _compute_q_phi_debug
        Q, D, Ydiff, S = _compute_q_phi_debug(Y, qt, 0.5, sumQ, 1)

        phi = Ydiff / (1 + np.expand_dims(D, 0))  # (2, 1000, 1000)
        EQy = ((np.expand_dims(Q, 0))*phi*S).sum(2)  # (2, 1000)
        dQy = -2*Q*(phi-2*np.expand_dims(EQy, 1))  # (2, 1000, 1000)
        dQx = (np.expand_dims(dYx, 2) * np.expand_dims(dQy, 3)).sum(0)

        dQxPhi = ((np.expand_dims(dQx, 0) * np.expand_dims(phi, 3))*np.expand_dims(S, axis=(0,3))).sum(2)

        phi = phi.transpose(1, 2, 0)
        dPhiy = np.expand_dims(np.eye(phi.shape[2], phi.shape[2]), axis=(0,1))-2*np.expand_dims(phi, 2)*np.expand_dims(phi, 3)
        dPhix = (np.expand_dims(dPhiy, 2) @ np.expand_dims(dYx.transpose(1,2,0), axis=(1,4))).squeeze()
        dPhixQ = ((dPhix.transpose(3,0,1,2) * np.expand_dims(Q, axis=(0,3))) * np.expand_dims(S, axis=(0,3))).sum(2)

        ddYx_longform2 = dQxPhi + dPhixQ

        self.assertTrue(np.isclose(ddYx_longform2, neg_array).all(),
                        'attribution from `_compute_attr_bh` does not match hand calculation in general')


if __name__ == '__main__':
    unittest.main()
