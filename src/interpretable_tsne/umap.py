# Author: Matthew Scicluna -- <mattcscicluna@gmail.com>
# Licence: BSD 3 clause (C) 2023

from .tsne import TSNE
from .tsne import _deprecate_positional_args

class UMAP(TSNE):
    r"""UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
    Abstract from the original paper [1]:
    UMAP is a novel manifold learning technique for dimension reduction. 
    UMAP is constructed from a theoretical framework based in Riemannian geometry and algebraic topology. 
    The result is a practical scalable algorithm that applies to real world data. 
    The UMAP algorithm is competitive with t-SNE for visualization quality, 
    and arguably preserves more of the global structure with superior run time performance. 
    Furthermore, UMAP has no computational restrictions on embedding dimension, 
    making it viable as a general purpose dimension reduction technique for machine learning.
    
    We exploit a connection between UMAP and t-SNE found in Böhm et al [2], which states that you
    can make embeddings that look very similar to UMAP by performing t-SNE with exaggeration term 
    (in addition to the usual early exaggeration done by t-SNE).
    
    For this approximation, a=b=1
    Need to figure out how to connect UMAP parameters to our sklearn t-SNE ones
    UMAP parameters here: https://umap-learn.readthedocs.io/en/latest/api.html
    specifically: n_neighbors=15, learning_rate=1.0

    Following [2], we set the early exaggeration to 12 and exaggeration to 4.
    
    Note that the connection between t-SNE and UMAP is further explored in Damrich et al. [3].

    ----------
    n_components : int, optional (default: 2)
        Dimension of the embedded space.
    perplexity : float, optional (default: 30)
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significanlty
        different results.
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
    [1] McInnes et al., (2018). UMAP: Uniform Manifold Approximation and Projection. 
        Journal of Open Source Software, 3(29), 861, 
        https://doi.org/10.21105/joss.00861
    [2] Jan Niklas Böhm, Philipp Berens, and Dmitry Kobak. (2022). 
        Attraction-repulsion spectrum in neighbor embeddings. 
        J. Mach. Learn. Res. 23, 1, Article 95.
    [3] Damrich, Sebastian and Fred A. Hamprecht (2021). 
        On UMAP's True Loss Function. 
        Neural Information Processing Systems.
    """
    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_N_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    @_deprecate_positional_args
    def __init__(self, n_components=2, *, perplexity=30.0,
                 learning_rate=200.0, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5,
                 attr='none', checkpoint_every=[], n_jobs=None):
        super().__init__(n_components=n_components, 
                         perplexity=perplexity,
                         early_exaggeration=12.0, 
                         exaggeration=1.0, 
                         learning_rate=learning_rate, 
                         n_iter=n_iter,
                         n_iter_without_progress=n_iter_without_progress, 
                         min_grad_norm=min_grad_norm,
                         metric=metric, 
                         init=init, 
                         verbose=verbose,
                         random_state=random_state, 
                         method='barnes_hut', 
                         angle=0.5,
                         attr=attr, 
                         checkpoint_every=checkpoint_every, 
                         n_jobs=n_jobs)
