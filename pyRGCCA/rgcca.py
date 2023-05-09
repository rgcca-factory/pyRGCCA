import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from numpy.linalg import norm, svd, inv
from scipy.linalg import sqrtm
from scipy.stats import ortho_group

from .deflate import deflate
from .utils import proj_l1_l2

class RGCCA(TransformerMixin, BaseEstimator):
    
    """Regularized Generalized Canonical Correlation Analysis (RGCCA)

    This class implements the block coordinate ascent (BCA) algorithm for 
    computing multiple components in a multiblock context. 
    
    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features_1, ..., n_features_J)]`.

    tau : list of shape (n_blocks,), default=[1]*n_blocks
        Regularization parameters for each block. Should be in [0, 1]
        for each block.

    connection : ndarray of shape (n_blocks, n_blocks), default=np.ones((n_blocks, n_blocks))-np.eye(n_blocks)
        Design matrix of the multiblock framework. Should be symetric with positive elements.

    scale : bool, default=True
        Whether to scale the blocks (w.r.t the mean and sd).

    scale_block : bool, default=True
        Whether to scale the blocks (w.r.t. the number of variables per block).

    scheme : string, default="factorial"
        The convex differentiable scheme function (g) to use for the criterion 

    init : string, default="random"
        The initialization procedure for the block weight vectors.

    max_iter : int, default=100
        The maximum number of iterations in the BCA algorithm.

    tol : float, default=1e-08
        The tolerance used as convergence criteria in BCA algorithm: the
        algorithm stops whenever the squared norm of any `v^k_j - v^k_{j-1}` 
        and `crit_j - crit_{j-1}` is less than `tol`, where `v^k` corresponds 
        to the k block weight vector and `crit` corresponds to the criterion.

    verbose : bool, default=True
        Whether to show the details during the execution and a summary when finished

    Attributes
    ----------
    scores_ : list of nd
    """
    def __init__(
        self,
        n_components=1,
        tau=None,
        connection=None,
        scale=True, 
        scale_block=True, 
        scheme='factorial',
        init='random',
        tol=1e-08,
        max_iter=100,
        verbose=True
    ):

        self.n_components = n_components
        self.connection = connection
        self.tau = tau
        self.scale = scale
        self.scale_block = scale_block
        self.scheme = scheme
        self.init = init
        self.tol = tol,
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y=None):

        J = len(X) # Number of blocks
        N = np.shape(X[0])[0] # Number of individuals
        p = [np.shape(block)[1] for block in X] # Number of variables per block
        C = self.connection
        M = [np.eye(p[j]) for j in range(J)]
        P = X.copy()

        if self.scheme == "factorial": g, dg = lambda x : x**2, lambda x : 2*x
        if self.scheme == "horst": g, dg = lambda x : x, lambda x : 1
        if self.scheme == "centroid": g, dg = lambda x : abs(x), lambda x : np.sign(x)

        self.scores_ = [np.zeros((N, self.n_components)) for _ in range(J)]
        self.loadings_ = [np.zeros((p[j], self.n_components)) for j in range(J)]
        self.loadings_true_ = [np.zeros((p[j], self.n_components)) for j in range(J)]
        self.n_features_ = [p[j] for j in range(J)]

        self._means = [np.zeros(p[j]) for j in range(J)]
        self._stds = [np.ones(p[j]) for j in range(J)]
        self._p = [np.zeros((p[j], self.n_components - 1)) for j in range(J)]
        
        min_ncomp = np.min(self.n_features_ + [N])
        if not 1 <= self.n_components <= min_ncomp:
            raise ValueError(f"n_components = {self.n_components} should be in [1, max(n_features_, N)] = [1, {min_ncomp}].")

        if self.scale:
            for j in range(J):
                self._means[j] = np.mean(X[j], axis=0)
                self._stds[j] = np.std(X[j], axis=0)
                X[j] = (X[j] - self._means[j]) / self._stds[j]

        if self.scale_block:
            for j in range(J):
                X[j] /= np.sqrt(p[j])

        # Defining constraint matrices
        for j in range(J):
            if self.tau[j] != 1:
                M[j] = self.tau[j] * np.eye(p[j]) + (1 - self.tau[j]) * (1 / (N - 1)) * X[j].T @ X[j]
            P[j] = (1/np.sqrt(N)) * X[j] @ inv(sqrtm(M[j]))

        Y = np.zeros((N, J))
        Z = np.zeros((N, J))

        for r in range(self.n_components):

            if self.verbose:
                print("\nComputation of the RGCCA block components #{} is under".format(str(r + 1)))

            # Deflation if compoment number greater than 1
            if r > 0:
                for j in range(J):
                    P[j], self._p[j][:,r-1] = deflate(P[j], v[j])

            v = []

            # Initializing
            if self.init == "random":
                for j in range(J):
                    v.append(np.random.normal(size=p[j]))
            if self.init == "svd":
                for j in range(J):
                    v.append(svd(P[j])[2][0,:])

            # Normalizing data & computing scores
            for j in range(J):
                v[j] = v[j] / np.sqrt(v[j].T @ M[j] @ v[j])
                Y[:,j] = P[j] @ v[j]

            iter = 1
            v_old = v.copy()
            crit_old = np.sum(C * g(Y.T @ Y))
            crit_list = [crit_old]

            while True:

                for j in range(J):
                    dgcov = dg(Y.T @ Y[:,j])
                    cdgcov = C[j,:] * dgcov
                    Z[:,j] = Y @ cdgcov
                    atmp = P[j].T @ Z[:,j] 
                    v[j] = atmp / norm(atmp)
                    Y[:,j] = P[j] @ v[j]

                crit = np.sum(C * g(Y.T @ Y))

                if self.verbose:
                    print(" Iter : {} Fit : {} Dif : {}".format(iter, crit, abs(crit - crit_old)))
                
                stopping_criteria = np.array(
                    [norm(np.concatenate([v_old[j] - v[j] for j in range(J)]))]
                    + [abs(crit - crit_old)]
                )

                if np.any(stopping_criteria < self.tol) or iter > self.max_iter:
                    break

                iter += 1
                v_old = v.copy()
                crit_old = crit
                crit_list.append(crit_old)

            for j in range(J):
                self.scores_[j][:,r] = P[j] @ inv(sqrtm(M[j])) @ v[j]
                self.loadings_[j][:,r] = inv(sqrtm(M[j])) @ v[j]
                self.loadings_true_[j][:,r] = inv(sqrtm(M[j])) @ v[j]
                # if n_components > 1 compute "true" loadings
                if r > 0:
                    self.loadings_true_[j][:,r] -= self.loadings_true_[j][:,:r] @ (inv(sqrtm(M[j])) @ v[j] @ self._p[j][:,:r])

        if self.verbose:
            plt.figure()
            plt.plot(crit_list)

        return self

    def transform(self, X, y=None):

        N = np.shape(X[0])[0]
        J = len(self.n_features_)

        scores = [np.zeros((N, self.n_components)) for _ in range(J)]

        for j in range(J):
            X[j] = (X[j] - self._means[j]) / self._stds[j]

        if self.scale_block:
            for j in range(J):
                X[j] = X[j] / np.sqrt(self.n_features_[j])

        for j in range(J):
            for r in range(self.n_components):
                scores[j][:,r] = X[j] @ self.loadings_true_[j][:,r]

        return scores

    def fit_transform(self, X, y=None):

        return self.fit(X, y).transform(X, y)

class GlobalRGCCA(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        n_components=1,
        tau=None,
        scale=True, 
        scale_block=True, 
        connection=None,
        scheme='factorial',
        init='random',
        tol=1e-8,
        max_iter=100,
        verbose=True
    ):
        self.scale = scale
        self.scale_block = scale_block
        self.connection = connection
        self.scheme = scheme
        self.n_components = n_components
        self.tau = tau
        self.init = init
        self.tol = tol,
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y=None):

        J = len(X) # Number of blocks
        N = np.shape(X[0])[0] # Number of individuals
        p = [np.shape(block)[1] for block in X] # Number of variables per block
        C = self.connection
        M = [np.eye(p[j]) for j in range(J)]
        P = X.copy()

        if self.scheme == "factorial": g, dg = lambda x : x**2, lambda x : 2*x
        if self.scheme == "horst": g, dg = lambda x : x, lambda x : 1
        if self.scheme == "centroid": g, dg = lambda x : abs(x), lambda x : np.sign(x)

        self.scores_ = np.zeros((N, self.n_components, J))
        self.loadings_ = [np.zeros((p[j], self.n_components)) for j in range(J)]

        if self.scale:
            for j in range(J):
                X[j] = (X[j] - np.mean(X[j], axis=0)) / np.std(X[j], axis=0)

        if self.scale_block:
            for j in range(J):
                X[j] /= np.sqrt(p[j])

        # Defining constraint matrices
        for j in range(J):
            if self.tau[j] != 1:
                M[j] = self.tau[j] * np.eye(p[j]) + (1 - self.tau[j]) * (1 / (N - 1)) * X[j].T @ X[j]
            P[j] = (1/np.sqrt(N)) * X[j] @ inv(sqrtm(M[j]))

        V = [np.zeros((p[j], self.n_components)) for j in range(J)]
        Y = np.zeros((N, self.n_components, J))
        Z = np.zeros((N, self.n_components, J))

        # Initializing
        if self.init == "random":
            for j in range(J):
                V[j] = ortho_group.rvs(p[j])[:, :self.n_components]
        if self.init == "svd":
            for j in range(J):
                V[j] = svd(X[j])[2][:self.n_components,:].T

        for j in range(J):
            Y[:,:,j] = P[j] @ V[j]

        iter = 1
        V_old = V.copy()
        crit_old = np.sum(C * np.array([[np.trace(g(Y[:,:,j].T @ Y[:,:,k])) for j in range(J)] for k in range(J)]))
        crit_list = []

        while True:

            for j in range(J):
                dgcov = [dg(Y[:,r,:].T @ Y[:,r,j]) for r in range(self.n_components)]
                cdgcov = [C[j,:] * dgcov[r] for r in range(self.n_components)]
                Z[:,:,j] = np.array([Y[:,r,:] @ cdgcov[r] for r in range(self.n_components)]).T
                grad = P[j].T @ Z[:,:,j]
                Q, _, R = svd(grad)
                V[j] = Q[:, :self.n_components] @ R[:self.n_components, :]
                Y[:,:,j] = P[j] @ V[j]

            crit = np.sum(C * np.array([[np.trace(g(Y[:,:,j].T @ Y[:,:,k])) for j in range(J)] for k in range(J)]))

            if self.verbose:
                print(" Iter : {} Fit : {} Dif : {}".format(iter, crit, abs(crit - crit_old)))
            
            stopping_criteria = np.array(
                [norm(np.concatenate([V_old[j] - V[j] for j in range(J)]))]
                + [abs(crit - crit_old)]
            )

            if np.any(stopping_criteria < self.tol) or iter > self.max_iter:
                break

            iter += 1
            V_old = V.copy()
            crit_old = crit
            crit_list.append(crit_old)

            for j in range(J):
                self.scores_[:,:,j] = Y[:,:,j]
                self.loadings_[j] = inv(sqrtm(M[j])) @ V[j]

        if self.verbose:
            plt.figure()
            plt.plot(crit_list)

        return self

    def fit_transform(self, X, y=None):

        return self

class SGCCA(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        n_components=1,
        sparsity=None,
        scale=True, 
        scale_block=True, 
        connection=None,
        scheme='factorial',
        init='random',
        tol=1e-8,
        max_iter=100,
        verbose=True
    ):
        self.scale = scale
        self.scale_block = scale_block
        self.connection = connection
        self.scheme = scheme
        self.n_components = n_components
        self.sparsity = sparsity
        self.init = init
        self.tol = tol,
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y=None):

        J = len(X) # Number of blocks
        N = np.shape(X[0])[0] # Number of individuals
        p = [np.shape(block)[1] for block in X] # Number of variables per block
        C = self.connection
        P = X.copy()

        if self.scheme == "factorial": g, dg = lambda x : x**2, lambda x : 2*x
        if self.scheme == "horst": g, dg = lambda x : x, lambda x : 1
        if self.scheme == "centroid": g, dg = lambda x : abs(x), lambda x : np.sign(x)

        self.scores_ = [np.zeros((N, self.n_components)) for _ in range(J)]
        self.loadings_ = [np.zeros((p[j], self.n_components)) for j in range(J)]
        self.loadings_true_ = [np.zeros((p[j], self.n_components)) for j in range(J)]
        self.n_features_ = [p[j] for j in range(J)]

        self._means = [np.zeros(p[j]) for j in range(J)]
        self._stds = [np.ones(p[j]) for j in range(J)]
        self._p = [np.zeros((p[j], self.n_components - 1)) for j in range(J)]
        
        min_ncomp = np.min(self.n_features_ + [N])
        if not 1 <= self.n_components <= min_ncomp:
            raise ValueError(f"n_components = {self.n_components} should be in [1, max(n_features_, N)] = [1, {min_ncomp}].")

        for j in range(J):
            if self.sparsity[j] < (1 / np.sqrt(p[j])):
                raise ValueError(f"sparsity for block {j} = {self.sparsity[j]} should be greater than 1/sqrt(p[j]) = {(1/np.sqrt(p[j]))}")
            else:
                self.sparsity[j] *= np.sqrt(p[j])

        if self.scale:
            for j in range(J):
                self._means[j] = np.mean(X[j], axis=0)
                self._stds[j] = np.std(X[j], axis=0)
                X[j] = (X[j] - self._means[j]) / self._stds[j]

        if self.scale_block:
            for j in range(J):
                X[j] /= np.sqrt(p[j])

        # Defining constraint matrices
        for j in range(J):
            P[j] = (1/np.sqrt(N)) * X[j]

        Y = np.zeros((N, J))
        Z = np.zeros((N, J))

        for r in range(self.n_components):

            if self.verbose:
                print("\nComputation of the RGCCA block components #{} is under".format(str(r + 1)))

            # Deflation if compoment number greater than 1
            if r > 0:
                for j in range(J):
                    P[j], self._p[j][:,r-1] = deflate(P[j], v[j])

            v = []

            # Initializing
            if self.init == "random":
                for j in range(J):
                    v.append(np.random.normal(size=p[j]))
            if self.init == "svd":
                for j in range(J):
                    v.append(svd(P[j])[2][0,:])

            # Normalizing data
            for j in range(J):
                v[j] = v[j] / norm(v[j])
                Y[:,j] = P[j] @ v[j]

            iter = 1
            v_old = v.copy()
            crit_old = np.sum(C * g(Y.T @ Y))
            crit_list = [crit_old]

            while True:

                for j in range(J):
                    dgcov = dg(Y.T @ Y[:,j])
                    cdgcov = C[j,:] * dgcov
                    Z[:,j] = Y @ cdgcov
                    grad = P[j].T @ Z[:,j]
                    v[j] = proj_l1_l2(grad, self.sparsity[j])
                    Y[:,j] = P[j] @ v[j]

                crit = np.sum(C * g(Y.T @ Y))

                if self.verbose:
                    print(" Iter : {} Fit : {} Dif : {}".format(iter, crit, crit - crit_old))
                
                stopping_criteria = np.array(
                    [norm(np.concatenate([v_old[j] - v[j] for j in range(J)]))]
                    + [abs(crit - crit_old)]
                )

                if np.any(stopping_criteria < self.tol) or iter > self.max_iter:
                    break

                iter += 1
                v_old = v.copy()
                crit_old = crit
                crit_list.append(crit_old)

            for j in range(J):
                self.scores_[j][:,r] = P[j] @ v[j]
                self.loadings_[j][:,r] = v[j]
                self.loadings_true_[j][:,r] = v[j]
                # if n_components > 1 compute "true" loadings
                if r > 0:
                    self.loadings_true_[j][:,r] -= self.loadings_true_[j][:,:r] @ (v[j] @ self._p[j][:,:r])

        if self.verbose:
            plt.figure()
            plt.plot(crit_list)

        return self

    def fit_transform(self, X, y=None):

        return self
