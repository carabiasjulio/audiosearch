from utils import *
from sklearn import neighbors
import numpy as np


class SearchModel(object):
    def __init__(self):
        """ libsamples_path: path to .npy file that loads to a numpy array of shape (n,d). """
        # model states
        self.libsamples = X
        N,D = X.shape
        self.scores = np.zeros((N,))
        self.feedback = ([],[])           # (indices_true, indices_false)  
        self.query_examples = np.zeros((0,D))    # key: file name    value: feature vector
        self.learner = neighbors.KNeighborsClassifier()

    def add_example(self, f):
        """ f: string, audio file name """
        x = get_features(f)
        self.query_examples[f] = x  # use hash(f) as key instead?

    def remove_example(self, f):
        self.query_examples.pop(f)

    def add_feedback(self, label_class, s_ind):
        ''' label_class: boolean
            s_ind: sample index 
        '''
        self.feedback[label_class].append(s_ind)

    def remove_feedback(self, label_class, s_ind):
        ''' label_class: boolean
            s_ind: sample index 
        '''
        self.feedback[label_class].remove(s_ind)

    def update_scores(self):
        """ re-rank samples in search_scope based on query examples and current feedback. Update self.ranking . """
        print 're-score'
        # traing samples
        I0, I1 = self.feedback
        print I0
        print self.libsamples[I0]

        X1 = np.concatenate((self.query_examples.values(), self.libsamples[I1]))    
        if len(X1)==0:
            print 'No query examples. Abort'
            return
        X0 = self.libsamples[I0]
        # unknown samples
        Ix = np.ones((N,))
        Ix[I0+I1]=0
        L = self.libsamples[Ix]

        # score by mean distance ratio
        m = neighbors.DistanceMetric.get_metric('minkowski')
        # mean distance to positive samples
        D1 = m.pairwise(L, X1).sum(axis=1) 
        if len(X0)==0:
            D0 = 1
        else:
            # mean distance to negatice samples
            D0 = m.pairwise(L, X0).sum(axis=1)
        self.scores = np.divide(D0,D1)  # higher is better

