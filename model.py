from utils import *
from sklearn import neighbors
import numpy as np


class SearchModel(object):
    def __init__(self):
        # model states
        self.scores = np.random.rand(N)     # random score upon start (thus user sees random samples at first)
        self.feedback = (np.zeros(N, dtype=bool), np.zeros(N, dtype=bool))      # (indices_true, indices_false)  
        self.examples = np.zeros((0,D))    # never delete loaded examples; should use database in practice 
        self.example_files = []
        self.example_active = []     # indices of active query examples i.e. deleted
        self.learner = neighbors.KNeighborsClassifier()

    def add_example(self, f):
        """ f: string, audio file name """
        x = get_features(f)
        self.example_active.append(len(self.examples))
        self.examples = np.concatenate((self.examples, x.reshape((1,D))))
        self.example_files.append(f)

    def remove_example(self, f):
        f_ind = self.example_files.index(f)
        self.example_active.remove(f_ind)

    def add_feedback(self, class_label, s_ind):
        ''' class_label: boolean
            s_ind: sample index 
        '''
        self.feedback[class_label][s_ind]=True

    def remove_feedback(self, class_label, s_ind):
        ''' class_label: boolean
            s_ind: sample index 
        '''
        self.feedback[class_label][s_ind]=False

    def remove_all_feedback(self, class_label):
        F = len(self.feedback[class_label])
        [self.feedback[class_label].pop() for i in range(F)]

    def update_scores(self, score):
        """ re-rank samples in search_scope based on query examples and current feedback. Update self.ranking . 
        score: function with signature score(E, I0, I1) = scores where scores is an numpy array of shape (N,) """

        print 're-score'
        X0, X1, L = self.get_learning_data()
        self.scores = score(X0, X1, L)

    def get_learning_data(self):
        I0,I1,Ix = self.get_index_partition()
        # labeled samples
        print 'feedback indices', I0.nonzero(), I1.nonzero()
        print 'examples', self.examples.shape, self.example_active
        X1 = np.concatenate((self.examples[self.example_active], X[I1]))    
        X0 = X[I0]
        if len(X1)+len(X0)==0:
            print 'No query inputs. Abort'
            return
        # unlabeled samples
        L = X[Ix]
        return X0, X1, L

    def get_index_partition(self):
        I0, I1 = self.feedback
        Ix = np.invert(np.any((I0, I1), axis=0))
        return I0, I1, Ix

    def get_proposals(self, n_proposal):
        I0, I1, Ix = self.get_index_partition()
        sort_inds = np.argsort(self.scores)[:-n_proposal-1:-1]  # numpy sort ascendingly; select from back
        # convert to library index
        raw_inds = np.flatnonzero(Ix)[sort_inds]
        #TODO: pass/use scores 
        return zip(LIBSAMPLE_PATHS[raw_inds], raw_inds, self.scores[sort_inds])


def mean_dist_ratio(X0, X1, L):
    """ 
    score by mean distance ratio
    X0: positive samples, shape (n0,d)
    X1: negative samples, shape (n1,d)
    L: unlabeled samples, shape (l,d)
    return: 1-D array of length l   scores of unlabeled library samples
    """
    m = neighbors.DistanceMetric.get_metric('minkowski')
    # mean distance to positive samples
    D1 = m.pairwise(L, X1).sum(axis=1) 
    if len(X0)==0:
        D0 = 1
    else:
        # mean distance to negatice samples
        D0 = m.pairwise(L, X0).sum(axis=1)
    scores = np.divide(D0,D1)  # higher is better; scores excludes feedback samples
    return scores

def p_knn(X0, X1, L):
    """
    score as class ratio among k nearest neighbors (class probability)
    X0: positive samples, shape (n0,d)
    X1: negative samples, shape (n1,d)
    L: unlabeled samples, shape (l,d)
    return: 1-D array of length l   scores of unlabeled library samples
    """
    n0 = len(X0)
    n1 = len(X1)
    if not n0:
        raise Exception('Need negative samples.')
    if not n1:
        raise Exception('Need positive samples.')
    k = np.min((12, (len(X0)+len(X1))/5))
    if k==0:
        print 'not enough data for k nearest neighbor; setting k=1'
        k=1

    c = neighbors.KNeighborsClassifier(n_neighbors=k)
    c.fit(np.concatenate((X0,X1)), np.concatenate((np.zeros(n0), np.ones(n1))) )
    scores = c.predict_proba(L)[:,1]
    return scores
