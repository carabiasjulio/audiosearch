from utils import *
from sklearn import neighbors, naive_bayes
import numpy as np
import logging

### Different score functions

def mean_dist_ratio(X0, X1, L):
    """ 
    score by mean distance ratio
    X0: positive samples, shape (n0,d)
    X1: negative samples, shape (n1,d)
    L: unfeedback_classed samples, shape (l,d)
    return: 1-D array of length l   scores of unfeedback_classed library samples
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
    L: unfeedback_classed samples, shape (l,d)
    return: 1-D array of length l   scores of unfeedback_classed library samples
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
    L: unfeedback_classed samples, shape (l,d)
    return: 1-D array of length l   scores of unfeedback_classed library samples
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

SCORE_FUNCS = [mean_dist_ratio, p_knn, p_MNB]
FEEDBACK_LABELS= ['negative', 'positive']

class SearchModel(object):
    def __init__(self, user):
        # model states
        self.scores = np.random.rand(N)     # random score upon start (thus user sees random samples at first)
        self.feedback = (np.zeros(N, dtype=bool), np.zeros(N, dtype=bool))      # (indices_true, indices_false)  
        self.examples = np.zeros((0,D))    # never delete loaded examples; should use database in practice 
        self.example_files = []
        self.example_active = []     # indices of active query examples i.e. deleted
#        self.score_func = mean_dist_ratio
        self.target_class = None    # target retrieval class
        self.target_example = -1 
        self.user = user
        logging.basicConfig(filename=user+'.log', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        logging.info('\nNEW TASK')
    
    def restart(self):
        self.__init__(self.user)

    def set_target_class(self, target):
        self.target_class = target
        logging.info("Target class set to #%d %s" %(target, CLASS_NAMES[target]))
    
    def task_completed(self):
        """ Task is to find n_goal target samples """
        n_goal = 10
        return len(self.get_feedback(1))>=n_goal or self.target_class<0

    def get_target_example(self):
        s_ind = np.random.choice(np.flatnonzero(Y==self.target_class)) 
        self.target_example = s_ind
        return (s_ind, LIBSAMPLE_PATHS[s_ind])
    
    def get_random_class_sample(self):
        if self.target_class!=None:
            logging.info("Retrieved a random sample of sound class #%d %s" % (self.target_class, CLASS_NAMES[self.target_class]))
            s_ind = np.random.choice(np.flatnonzero(Y==self.target_class))
            return (LIBSAMPLE_PATHS[s_ind], s_ind)
        else:
            raise Exception("Target class not set!")


    def add_example(self, f):
        """ f: string, audio file name """
        x = get_features(f)
        self.example_active.append(len(self.examples))
        self.examples = np.concatenate((self.examples, x.reshape((1,D))))
        self.example_files.append(f)
        logging.info("Added example %s"%f)

    def remove_example(self, f):
        f_ind = self.example_files.index(f)
        self.example_active.remove(f_ind)
        logging.info("Removed example %s"%f)

    def add_feedback(self, feedback_class, s_ind):
        ''' feedback_class: boolean
            s_ind: int or 1-D array, sample index 
        '''
        self.feedback[feedback_class][s_ind]=True
        logging.info("Labeled sample %s as %s"%(s_ind, FEEDBACK_LABELS[feedback_class]))

    def remove_feedback(self, feedback_class, s_ind):
        ''' feedback_class: boolean
            s_ind: sample index 
        '''
        # TODO: does not need feedback_class lol
        self.feedback[feedback_class][s_ind]=False
        logging.info("Unlabeled sample %d" % s_ind)

    def remove_all_feedback(self, feedback_class):
        self.feedback[feedback_class][:]=False
        logging.info("Unlabeled all %s samples" % FEEDBACK_LABELS[feedback_class])

    def get_feedback(self, feedback_class):
        I = np.flatnonzero(self.feedback[feedback_class])
        return zip(LIBSAMPLE_PATHS[I], I)

    def update_scores(self, k, weighted, metric):
        """ re-score samples in LIBSAMPLE_PATHS based on query examples and current feedback. Update self.scores. 
        score: function f with signature f(E, I0, I1) = scores where scores is an numpy array of shape (N,) """
        print 're-score'
        X0, X1, L = self.get_learning_data()
        if len(X1)==0:
            print 'No positive examples. Abort'
            return
        if len(X0)==0:
            score_func = mean_dist_ratio
            print 'No negative examples. Ranking samples based on distance to positive examples.'
        else:
            c = neighbors.KNeighborsClassifier(n_neighbors=k, weights=['uniform', 'distance'][weighted], metric=metric, p=10)
            score_func = p_classifier(c)
        self.scores = score_func(X0, X1, L)
        logging.info("Score updated")

    def get_learning_data(self):
        I0,I1,Ix = self.get_index_partition()
        # feedback_classed samples
        print 'feedback indices', I0.nonzero(), I1.nonzero()
        print 'examples', self.examples.shape, self.example_active
        X1 = np.concatenate((self.examples[self.example_active], X[I1]))    
        X0 = X[I0]
        # unfeedback_classed samples
        L = X[Ix]

        return X0, X1, L

    def get_trainset_size(self):
        I0, I1 = self.feedback
        return I0.sum()+I1.sum()+(self.target_example>=0)+len(self.example_active)

    def get_index_partition(self):
        I0, I1 = self.feedback
        # Use copy
        I1 = I1.copy()
        # incorporate target example 
        if self.target_example>=0:
            I1[self.target_example]=1
        Ix = np.invert(np.any((I0, I1), axis=0))
        return I0, I1, Ix

    def get_proposals(self, n_proposal):
        I0, I1, Ix = self.get_index_partition()
        sort_inds = np.argsort(self.scores)[:-n_proposal-1:-1]  # numpy sort ascendingly; select from back
        # convert to library index
        raw_inds = np.flatnonzero(Ix)[sort_inds]
        #TODO: pass/use scores 
        logging.info("Proposing %s \n %s" % (raw_inds, LIBSAMPLE_PATHS[raw_inds])) 
        return zip(LIBSAMPLE_PATHS[raw_inds], raw_inds, self.scores[sort_inds])

    def get_random_samples(self, n_samples):
        _,_,Ix = self.get_index_partition()
        raw_inds = np.flatnonzero(Ix)
        np.random.shuffle(raw_inds)
        return zip(LIBSAMPLE_PATHS[raw_inds[:n_samples]], raw_inds[:n_samples])


