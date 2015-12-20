from utils import *
from sklearn import neighbors
import numpy as np


class SearchModel(object):
    def __init__(self):
        """ libsamples_path: path to .npy file that loads to a numpy array of shape (n,d). """
        # model states
        self.libsamples = X
        N,D = X.shape
        self.N = N
        self.D = D
        self.scores = np.zeros((N,))
        self.feedback = ([],[])           # (indices_true, indices_false)  
        self.examples = np.zeros((0,D))    # never delete loaded examples; should use database in practice 
        self.example_files = []
        self.example_active = []     # indices of active query examples i.e. deleted
        self.learner = neighbors.KNeighborsClassifier()

    def add_example(self, f):
        """ f: string, audio file name """
        x = get_features(f)
        self.example_active.append(len(self.examples))
        self.examples = np.concatenate((self.examples, x.reshape((1,self.D))))
        self.example_files.append(f)

    def remove_example(self, f):
        f_ind = self.example_files.index(f)
        self.example_active.remove(f_ind)

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
        print 'feedback inds', I0, I1
        print 'examples', self.examples.shape, self.example_active
        X1 = np.concatenate((self.examples[self.example_active], self.libsamples[I1]))    
        if len(X1)==0:
            print 'No query examples. Abort'
            return
        X0 = self.libsamples[I0]
        # unknown samples
        Ix = np.ones((N,), dtype=bool)
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

