'''
This unittest tests the Barnes-Hut approximation of the t-SNE gradient computations.
We set a threshold of 95% Spearman Correlation between exact and approximate P and dP tensors.
We set threshold of 97% Spearman Correlation between exact and approximate Attributions and Embedding values
'''

import unittest
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
import numpy as np

#from _grad_comps import *
from interpretable_tsne.tsne import pairwise_distances, _joint_probabilities, _compute_dp, _openmp_effective_n_threads, _joint_probabilities_nn, _compute_dp_bh, NearestNeighbors, TSNE
# src.interpretable_tsne ...

def get_dPs(data, perplexity=30):
    # helper function that computes P, dP for regular and barnes-Hut
    n_samples = data.shape[0]
    distances = pairwise_distances(data, metric="euclidean", squared=True)
    P, conditional_P, betas = _joint_probabilities(distances, perplexity, verbose=0)
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
                                                      0)

    # compute dP
    _, dP2 = _compute_dp_bh(data,
                           conditional_P2,
                           P2,
                           betas2,
                           max(2 - 1, 1),  # degrees of freedom (see below for
                           _openmp_effective_n_threads())
    

    dense_mats = [csr_matrix((dP2[:,i], P2.indices, P2.indptr), shape=(n_samples, n_samples)).toarray() for i in range(data.shape[1])]
    dP2 = np.stack(dense_mats)
    dP2 = dP2.transpose(1, 2, 0)
    #dP2 = dP2.reshape(n_samples, n_samples, data.shape[1])
    return dP, dP2, P, P2


def run_tsne_reproducible(data, seed_value, method):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value

    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    tsne = TSNE(method=method, attr='grad_norm', verbose=0, n_iter=251, early_exaggeration=4.0, learning_rate=200.0, checkpoint_every=[10,20,50,100,249])
    return_obj = tsne.fit_transform(data)
    return return_obj['embeddings'], return_obj['attrs']


class Test_bh_approx_P_dP(unittest.TestCase):

    def setUp(self):

        # for reproducibility
        np.random.seed(seed=42)

        # load sample MNIST data
        file = np.load('tests/test_data.npz', allow_pickle=True)
        X = file['data']
        
        # Compute the affinities matrix with and without Barnes-Hut approximation
        dP, dP2, P, P2 = get_dPs(X, perplexity=30)
        
        self.dP = dP
        self.dP2 = dP2
        self.P = P
        self.P2 = P2

    def test_P(self):
        # Format this into an array
        P = squareform(self.P).flatten()
        P2 = np.array(self.P2.todense()).flatten()

        # Ignore 0 values
        P = P[P2 > 0]
        P2 = P2[P2 > 0]

        # we dont expect these to be the same, so lets say that 95% similarity is close enough!
        spear_P = spearmanr(P, P2).correlation
        assert spear_P > 0.97, 'Barnes-Hut approximation of `P` is too different from original (Spearman r={:.2f})'.format(spear_P)

    def test_dP(self):
        for i in range(50):
            # Format this into an array
            dP = self.dP[:,:,i].flatten()
            dP2 = self.dP2[:,:,i].flatten()

            # Ignore 0 values
            dP = dP[dP2 > 0]
            dP2 = dP2[dP2 > 0]

            # we dont expect these to be the same, so lets say that 95% similarity is close enough!
            spear_P = spearmanr(dP, dP2).correlation
            assert spear_P > 0.95, 'Barnes-Hut approximation of `dP` (for feature {}) is too different from original (Spearman r={:.2f})'.format(i, spear_P)


class Test_bh_approx_emb_attrs(unittest.TestCase):

    def setUp(self):

        # for reproducibility
        np.random.seed(seed=42)

        # load sample MNIST data
        file = np.load('tests/test_data.npz', allow_pickle=True)
        X = file['data']

        #  Need to subset when computing exact t-SNE since it is very slow!
        r_idx = np.random.choice(X.shape[0], 500, replace=False)
        X = X[r_idx, :10]

        emb, attr = run_tsne_reproducible(X, 20, 'exact')
        emb2, attr2 = run_tsne_reproducible(X, 20, 'barnes_hut')
        self.emb = emb
        self.emb2 = emb2
        self.attr = attr
        self.attr2 = attr2

    def test_embs_attrs(self):
        for t in range(2, len(self.emb)): # ignore first two since BH bad approx for uniform data
            spear_P = spearmanr(self.emb[t].flatten(), self.emb2[t].flatten()).correlation
            assert spear_P > 0.97, 'Barnes-Hut approximation of `Embeddings` at iter {} is too different from Exact (Spearman r={:.2f})'.format(t,
            spear_P)

        for t in range(2, len(self.attr)): # ignore first two since BH bad approx for uniform data
            attr_t = self.attr[t]
            attr2_t = self.attr2[t]
            no_nan = (~np.isnan(attr_t)) & (~np.isnan(attr2_t)) # ignore values that are very small
            spear_P = spearmanr(attr_t[no_nan].flatten(),attr2_t[no_nan].flatten()).correlation
            assert spear_P > 0.97, 'Barnes-Hut approximation of `attributions` at iter {} is too different from Exact (Spearman r={:.2f})'.format(t, spear_P)


if __name__ == '__main__':
    unittest.main()
