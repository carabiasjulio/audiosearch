from utils import *
from sklearn import neighbors, naive_bayes
import numpy as np

### Different score functions

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
#    k = np.min((12, (n0+n1)/5))
#    if k==0:
#        print 'not enough data for k nearest neighbor; setting k=1'
#        k=1

    c = neighbors.KNeighborsClassifier()
    c.fit(np.concatenate((X0,X1)), np.concatenate((np.zeros(n0), np.ones(n1))) )
    scores = c.predict_proba(L)[:,1]
    return scores

def p_nc(X0, X1, L):
    """
    Nearest centroid classifer. 
    Score is class prediction. (1 or 0)
    X0: positive samples, shape (n0,d)
    X1: negative samples, shape (n1,d)
    L: unlabeled samples, shape (l,d)
    return: 1-D array of length l   scores of unlabeled library samples
    """
    n0, n1 = map(len, [X0,X1])
    c = neighbors.NearestCentroid(shrink_threshold=0.1) #metric='minkowski')
    c.fit(np.concatenate((X0,X1)), np.concatenate((np.zeros(n0), np.ones(n1))) )
    scores = c.predict(L)  # does not have score by default
    print scores
    scores[np.array(scores, dtype=bool)] = np.random.rand(scores.sum())   # randomize predicted positives
    return scores


def p_classifier( c):
    """ return a score function """
    def f(X0, X1, L):
        n0 = len(X0)
        n1 = len(X1)
        if not n0:
            raise Exception('Need negative samples.')
        if not n1:
            raise Exception('Need positive samples.')
    #    k = np.min((12, (n0+n1)/5))
    #    if k==0:
    #        print 'not enough data for k nearest neighbor; setting k=1'
    #        k=1

        c.fit(np.concatenate((X0,X1)), np.concatenate((np.zeros(n0), np.ones(n1))) )
        scores = c.predict_proba(L)[:,1]
        return scores
    return f

p_MNB = p_classifier(naive_bayes.MultinomialNB())
p_GNB = p_classifier(naive_bayes.GaussianNB())

class SearchModel(object):
    def __init__(self):
        # model states
        self.scores = np.random.rand(N)     # random score upon start (thus user sees random samples at first)
        self.feedback = (np.zeros(N, dtype=bool), np.zeros(N, dtype=bool))      # (indices_true, indices_false)  
        self.examples = np.zeros((0,D))    # never delete loaded examples; should use database in practice 
        self.example_files = []
        self.example_active = []     # indices of active query examples i.e. deleted
        self.score_func = mean_dist_ratio

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
            s_ind: int or 1-D array, sample index 
        '''
        self.feedback[class_label][s_ind]=True
        print 'added feedback:', class_label, s_ind 

    def remove_feedback(self, class_label, s_ind):
        ''' class_label: boolean
            s_ind: sample index 
        '''
        # TODO: does not need class_label lol
        self.feedback[class_label][s_ind]=False
        print 'removed feedback', class_label, s_ind

    def remove_all_feedback(self, class_label):
        self.feedback[class_label][:]=False

    def get_feedback(self, class_label):
        I = np.flatnonzero(self.feedback[class_label])
        return zip(LIBSAMPLE_PATHS[I], I)

    def update_scores(self, score=None):
        """ re-score samples in LIBSAMPLE_PATHS based on query examples and current feedback. Update self.scores. 
        score: function f with signature f(E, I0, I1) = scores where scores is an numpy array of shape (N,) """
        if not score:
            score = self.score_func
        print 're-score'
        X0, X1, L = self.get_learning_data()
        if len(X1)==0:
            print 'No positive examples. Abort'
            return
        self.scores = score(X0, X1, L)

    def get_learning_data(self):
        I0,I1,Ix = self.get_index_partition()
        # labeled samples
        print 'feedback indices', I0.nonzero(), I1.nonzero()
        print 'examples', self.examples.shape, self.example_active
        X1 = np.concatenate((self.examples[self.example_active], X[I1]))    
        X0 = X[I0]
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

    def get_random_samples(self, n_samples):
        _,_,Ix = self.get_index_partition()
        raw_inds = np.flatnonzero(Ix)
        np.random.shuffle(raw_inds)
        return zip(LIBSAMPLE_PATHS[raw_inds[:n_samples]], raw_inds[:n_samples])


