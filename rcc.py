import numpy as np
from numpy.linalg import norm, solve, lstsq

from scipy.spatial.distance import cdist, pdist
from scipy.spatial import distance
from scipy.sparse import csr_matrix, triu, find
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize, scale

import time
import scipy.sparse
import math

eps = 1e-5

class rcc_cluster():
    def __init__(self, k=10, verbose=True, preprocessing='none', measure='euclidean', clustering_threshold=1.):
        self.clustering_threshold = clustering_threshold
        self.preprocessing = preprocessing
        self.k = k
        self.verbose = verbose
        self.labels_ = []
        self.measure = measure

    def assignment(self, epsilon):
        diff = np.sum((self.U[self.i,:] - self.U[self.j,:])**2, axis=1)

        # computing connected components. 
        is_conn = np.sqrt(diff) <= self.clustering_threshold*epsilon

        G = scipy.sparse.coo_matrix((np.ones((2*np.sum(is_conn),)), (np.concatenate([self.i[is_conn], self.j[is_conn]], axis=0), np.concatenate([self.j[is_conn], self.i[is_conn]],axis=0))), shape=[self.nsamples,self.nsamples])

        num_components, labels = connected_components(G, directed=False)
        return labels, num_components

    def geman_mcclure(self, data, mu):
        return (mu / (mu + np.sum(data**2, axis=1)))**2

    def computeObj(self, X, U, lpq, i, j, landa, mu, weights, iter_num):

        # computing the objective as in equation [2]
        data = 0.5 * np.sum(np.sum((X-U)**2))
        diff = np.sum((U[i, :]-U[j,:])**2, axis=1)
        smooth = landa * 0.5 * (np.inner(lpq*weights, diff) + mu * np.inner(weights, (np.sqrt(lpq+eps)-1)**2))
            
        # final objective  
        obj = data + smooth
        if self.verbose:
            print(' {} | {} | {} | {}'.format(iter_num, data, smooth, obj))
    
        return obj

    def mkNN(self, X, k, measure='euclidean'):
        """
        This code is taken from:
        https://bitbucket.org/sohilas/robust-continuous-clustering/src/
        Construct mutual_kNN for large scale dataset

        If j is one of i's closest neighbors and i is also one of j's closest members,
        the edge will appear once with (i,j) where i < j.

        Parameters
        ----------
        X : [n_samples, n_dim] array
        k : int
          number of neighbors for each sample in X
        """

        samples = X.shape[0]
        batchsize = 10000
        b = np.arange(k+1)
        b = tuple(b[1:].ravel())

        z = np.zeros((samples,k))
        weigh = np.zeros_like(z)

        # This loop speeds up the computation by operating in batches
        # This can be parallelized to further utilize CPU/GPU resource

        for x in np.arange(0, samples, batchsize):
            start = x
            end = min(x+batchsize,samples)

            w = distance.cdist(X[start:end], X, measure)

            y = np.argpartition(w, b, axis=1)

            z[start:end,:] = y[:, 1:k + 1]
            weigh[start:end,:] = np.reshape(w[tuple(np.repeat(np.arange(end-start), k)), tuple(y[:, 1:k+1].ravel())], (end-start, k))
            del(w)

        ind = np.repeat(np.arange(samples), k)

        P = csr_matrix((np.ones((samples*k)), (ind.ravel(), z.ravel())), shape=(samples,samples))
        Q = csr_matrix((weigh.ravel(), (ind.ravel(), z.ravel())), shape=(samples,samples))

        Tcsr = minimum_spanning_tree(Q)
        P = P.minimum(P.transpose()) + Tcsr.maximum(Tcsr.transpose())
        P = triu(P, k=1)

        V = np.asarray(find(P)).T
        return V[:,:2].astype(np.int32)

    def RCC(self, X, w, maxiter=100, disp=True, inner_iter=4):

        count = 0

        X = X.astype(np.float32) # features stacked as N x D (D is the dimension)
        assert len(X.shape) == 2

        w = w.astype(np.int32) # list of edges represented by start and end nodes
        assert w.shape[1] == 2

        i = w[:, 0]
        j = w[:, 1]

        # initialization
        nsamples, nfeatures = X.shape

        npairs = w.shape[0]

        # precomputing xi
        xi = np.linalg.norm(X, 2)

        # setting weights as given in equation [S1]
        R = scipy.sparse.coo_matrix((np.ones((i.shape[0]*2,)), 
            (np.concatenate([i, j], axis=0), np.concatenate([j, i],axis=0))), shape=[nsamples, nsamples])

        nconn = np.sum(R, axis=1)
        nconn = np.asarray(nconn)

        weights = np.mean(nconn) / np.sqrt(nconn[i]*nconn[j])
        weights = weights[:, 0]

        # initializing U to X and lpq = 1 forall p,q in E
        U = X.copy()
        lpq = np.ones((i.shape[0],))

        # computation of delta and mu
        epsilon = np.sqrt(np.sum((X[i, :] - X[j, :])**2 + eps, axis=1))

        # Note: suppress low values, but hard coded threshold has issues
        epsilon[epsilon/np.sqrt(nfeatures) < 1e-2] = np.max(epsilon)

        epsilon = np.sort(epsilon)

        mu = 3.0 * epsilon[-1]**2

        # top 1% of the closest neighbours
        top_samples = np.minimum(250.0, math.ceil(npairs*0.01)) 

        delta = np.mean(epsilon[:int(top_samples)])
        epsilon = np.mean(epsilon[:int(math.ceil(npairs*0.01))])

        # computation of matrix A = D-R (here D is the diagonal matrix and R is the symmetric matrix)
        R = scipy.sparse.coo_matrix((np.concatenate([weights*lpq, weights*lpq], axis=0), (np.concatenate([i,j],axis=0), np.concatenate([j,i],axis=0))), shape=[nsamples, nsamples])
        D = scipy.sparse.coo_matrix((np.squeeze(np.asarray(np.sum(R,axis=1))), ((range(nsamples), range(nsamples)))), (nsamples, nsamples))

        # initial computation of lambda (lambda is a reserved keyword in python)
        # note: the largest magnitude eigenvalue is equal to the L2 norm under certain conditions, and is faster to compute

        eigval = scipy.sparse.linalg.eigs(D-R, k=1, return_eigenvectors=False).real
        landa =  xi / eigval[0]

        if self.verbose:
            print('mu = {}, lambda = {}, epsilon = {}, delta = {}'.format(mu, landa, epsilon, delta))
            print(' Iter | Data \t | Smooth \t | Obj \t')    

        obj = np.zeros((maxiter,))

        # start of optimization phase
        starttime = time.time()

        for iter_num in range(1, maxiter):
            st = time.time()
            # update lpq
            lpq = self.geman_mcclure(U[i,:]-U[j,:], mu)

            # compute objective
            obj[iter_num] = self.computeObj(X, U, lpq, i, j, landa, mu, weights, iter_num)
            
            # update U's

            R = scipy.sparse.coo_matrix((np.concatenate([weights*lpq, weights*lpq], axis=0), (np.concatenate([i,j],axis=0), np.concatenate([j,i],axis=0))), shape=[nsamples, nsamples])

            D = scipy.sparse.coo_matrix((np.asarray(np.sum(R, axis=1))[:, 0], ((range(nsamples), range(nsamples)))), shape=(nsamples, nsamples))

            M = scipy.sparse.eye(nsamples) + landa * (D-R)
            U = scipy.sparse.linalg.spsolve(M,X)
          
            # check for stopping criteria
            count += 1

            if (abs(obj[iter_num-1]-obj[iter_num]) < 1e-1) or count == inner_iter:
                if mu >= delta:
                    mu /= 2.0
                elif count == inner_iter:
                    mu = 0.5 * delta
                else:
                    break

                eigval = scipy.sparse.linalg.eigs(D-R, k=1, return_eigenvectors=False).real
                landa =  xi / eigval[0]
                count = 0

        self.U = U.copy()
        self.i = i
        self.j = j
        self.nsamples = nsamples
        C, num_components = self.assignment(epsilon)

        return U, C


    def fit(self, X):

        n_samples, n_features = X.shape

        if self.preprocessing == 'scale':
            X = scale(X, copy=False)
        elif self.preprocessing == 'minmax':
            minmax_scale = MinMaxScaler().fit(X)
            X = minmax_scale.transform(X)
        elif self.preprocessing == 'normalization':
            X = np.sqrt(n_features) * normalize(X, copy=False)

        mknn_matrix = self.mkNN(X, self.k, measure=self.measure)
        U, C = self.RCC(X, mknn_matrix)
        self.labels_ = C.copy()

        return self.labels_

