from utils import *
from model import *
import datetime 
import matplotlib.pyplot as pl

TRAIN_SIZE = 40

def step(m, proposal_size):
    m.update_scores(min(3, m.get_trainset_size()))
    proposals = m.get_proposals(proposal_size)
    #print "new proposal classes:", Y[[zip(*proposals)[1]]]
    return proposals

def label_all(m, proposals, t):
    for s, ind, score in proposals:
        m.add_feedback(Y[ind]==t, ind)

def label_until_reject(m,proposals, t):
    for s, ind, score in proposals:
        if Y[ind]==t:
            m.add_feedback(1, ind)
        else:
            m.add_feedback(0, ind)
            break

def test_target(t, train_size, proposal_size=5, verbose=False, m_init=None):
    if m_init:
        m = m_init
    else:
        m = SearchModel('test'+str(t)+'_'+str(datetime.datetime.now()), mode = MODE_TEST)
        m.set_target_class(t)
        e, e_ind = m.get_random_class_sample()
        m.update_scores(1)

    proposals = step(m, proposal_size)
    while m.get_trainset_size()<=train_size:
#    for i in range(n_rounds):dd
#        label_all(proposals)
        label_until_reject(m, proposals, t)
        proposals = step(m, proposal_size)
    return m


def test_target_until_recall(t, recall, proposal_size=5, m_init=None):
    if m_init:
        m=m_init
    else:
        m = SearchModel('test', mode=MODE_TEST)
        m.set_target_class(t)
        q=m.get_random_class_sample()
    # number actual positives excluding query
    n = np.sum(Y==t)-1
    print n*recall 
    p = step(m, proposal_size)
    while m.feedback[1].sum()<n*recall:
        label_until_reject(m, p, t)
        p = step(m, proposal_size)
        r = m.feedback[1].sum(), m.get_trainset_size()
        print r
        yield r


def repeat_test_all(proposal_size, train_size=TRAIN_SIZE, repeats=20, verbose=False):
    rounds_per_test = train_size/proposal_size
    # Open label logs for all targets
    results = []
    for t in range(10):
        # accumulator
        try:
            results.append(np.load('trainset_c%d_%d_%d.npy'%(t, proposal_size, train_size)))
        except:
            results.append(np.empty((0,N)))

    # manual parallelization (lol) of repeated testing on all targets
    test_batchsize = 5  
    for batch in range(repeats/test_batchsize+1):
        for t in range(10):
            counts = repeat_test(t, proposal_size, train_size, test_batchsize, verbose)
            results[t] = np.concatenate((results[t], counts))
            np.save(open('trainset_c%d_%d_%d.npy'%(t, proposal_size, train_size), 'wb'), results[t])

    



def repeat_test(target, proposal_size,train_size = TRAIN_SIZE, repeats=20, verbose=False):
    rounds_per_test = train_size/proposal_size
    #F1_all = np.empty((repeats, rounds_per_test))
    I_train = np.empty((repeats,N))
    for i in range(repeats):
        if not (i+1)%10:
            print "iteration", i+1
        m = test_target(target, train_size, proposal_size, verbose)
        I_train[i]=np.any(m.feedback,0)

    return I_train


def plot_label_counts(F1, t, proposal_size, train_size=150):
    rounds_per_test = train_size/proposal_size
    F_total = np.arange(proposal_size, proposal_size, train_size+1)
    F0 = F_total - F1
    pl.plot(F1, 'g.', label='Accepted')
    pl.plot(F0,'r.', label='Rejected')
    pl.xlabel('iteration')
    pl.ylabel('number of labels')
    pl.xlim([0, rounds_per_test])
    pl.ylim([0,train_size])
    pl.title("Class %d (batchsize=%d)"%(t, proposal_size))
    pl.legend(loc='upper left')
    pl.show()


def plot_accuracy_for_proposal_size(t, b):
    F1 = np.load('c%d_%d.npy'%(t, b))
    pl.plot(np.divide(F1, np.arange(b,151,b)).mean(0), label='batchsize=%d'%b)
    

def plot_accuracy(t):
    pl.figure()
    plot_accuracy_for_proposal_size(t,3)
    plot_accuracy_for_proposal_size(t,5)
    plot_accuracy_for_proposal_size(t,10)
    pl.legend()
    pl.xlabel('round')
    pl.ylabel('proposal accuracy')
    pl.ylim([0,1])
    pl.title('Class %d: %s'%(t, CLASS_NAMES[t]))
    #pl.show()
    pl.savefig('accuracy_c%d.jpg'%t)



## Comparison with straigt-forward i.e. non-interactive training
trials = 20
PROPOSAL_SIZE= 5
target = 0 

def testset_accuracy_interactive(target, trials, proposal_size=5, train_size=TRAIN_SIZE):
    n_rounds = train_size/proposal_size
    """ Accuracy after interactive training"""
    results = []
    for i in range(trials):
        m = test_target(target, train_size, batch=proposal_size)
        _,_,test_ind = m.get_index_partition()
        train_ind = np.invert(test_ind)
        knn = neighbors.KNeighborsClassifier(min(12, train_size), 'distance')
        knn.fit(X[train_ind],Y[train_ind]==target)
        #accuracy = np.mean(knn.predict(X[test_ind])==(Y[test_ind]==target))
        ytrue = Y[test_ind]==target
        ypred = knn.predict(X[test_ind])
        recall = metrics.recall_score(ytrue, ypred)
        precision = metrics.precision_score(ytrue, ypred)
        auc = metrics.roc_auc_score(ytrue, ypred)
        pr = metrics.precision_recall_curve(ytrue, knn.predict_proba(X[test_ind])[:,1])
        results.append((recall, precision, auc,pr))
    return results 


def train_interact(target, train_size = TRAIN_SIZE, proposal_size =PROPOSAL_SIZE):
    m = test_target(target, train_size, proposal_size)
    try: 
        c = m.classifier

        _,_, i_test = m.get_index_partition()
        i_train = np.invert(i_test)
    #    c.fit(X[i_train], Y[i_train]==target)
        return c, i_test,m
    except:
        return train_interact(target, train_size, proposal_size)

def train_direct(target, train_size):
    c = neighbors.KNeighborsClassifier(max(2, train_size/3), 'distance')
    i_shuffled = range(N)
    np.random.shuffle(i_shuffled)
    i_train = i_shuffled[:train_size]
    i_test = i_shuffled[train_size:]
    c.fit(X[i_train], Y[i_train]==target)
    return c, i_test


def eval_interact(target, train_size, metric_func, repeats = 100,proposal_size=5):
    scores = []
    for i in range(repeats):
        m = test_target(target, train_size, proposal_size)
        scores.append(metric_func(m.scores, target, m.get_index_partition()[-1]))
    return scores

def eval_direct(target, train_size, metric_func, repeats=100):
    c = neighbors.KNeighborsClassifier(max(3, train_size/3), 'distance')
    i_shuffled = range(N)
    np.random.shuffle(i_shuffled)
    scores = []
    for i in range(repeats):
        i_train = i_shuffled[i*train_size:(i+1)*train_size]
        i_test = i_shuffled[:i*train_size]+i_shuffled[(i+1)*train_size:]
        c.fit(X[i_train], Y[i_train]==target)
        scores.append(metric_func(c, target, i_test))
    return scores

def precision_at(ypred, target, I_test, k):
    """ c: sklearn classifier (with function predict_proba)
        target: int, target class
        I_test: indices of test data
        k: 
        return: precision@k for classifier c on querying target in corpus indexed by I_test
    """
    ytrue = Y[I_test]==target
    i_sorted = np.argsort(ypred)[::-1]
    hits = np.sum(Y[I_test][i_sorted[:k]]==target)
    #print Y[I_test][i_sorted[:k]] 
    return 1.*hits/k

def r_precision(c, target, i_test):
    k = np.sum(Y[i_test]==target)
#    print k
    return precision_at(c, target, i_test, k)

def eval_interact_all(train_size, metric_func, repeats=50, proposal_size=5):
    S = np.empty((10, repeats))

    k = 20
    for t in range(10):
        S[t] = eval_interact(t, train_size, metric_func, repeats, proposal_size)
        np.save(open('S1_t%d_m%s_p%d.npy'%(train_size, metric_func, k),'wb'), S)
    return S

def eval_direct_all(train_size, metric_func, repeats=50):
    S = np.empty((10, repeats))
    for t in range(10):
        S[t] = eval_direct(t, train_size, metric_func, repeats)
        np.save(open('S0_t%d_M%.npy'%(train_size, metric_func),'wb'), S)
        #S[t] = eval_direct(t, train_size, lambda c,t,i: precision_at(c, t, i, k), repeats)
        #np.save(open('S0_t%d_k%d.npy'%(train_size,k),'wb'), S)

    return S

import sklearn.cross_validation as cv
import sklearn.metrics as metrics

def run_baseline_stratified(target, trials, train_size=TRAIN_SIZE):
    """ Accuracy after straigtforward training """
    # stratigied training set i.e. ~ 1/10 positive
    sss = cv.StratifiedShuffleSplit(Y==target, train_size=train_size, test_size=N-train_size, n_iter=trials)
    results = []
    for train_ind, test_ind in sss:
        x_train, y_train = X[train_ind], Y[train_ind]==target
        x_test, y_test = X[test_ind], Y[test_ind]==target
        knn = neighbors.KNeighborsClassifier(min(12, train_size),weights='distance')
        knn.fit(x_train, y_train)
        #accuracy = np.mean(knn.predict(x_test)==y_test)
#        recall = metrics.recall_score(y_test, knn.predict(x_test))
#        precision = metrics.precision_score(y_test, knn.predict(x_test))
#        auc = metrics.roc_auc_score(y_test, knn.predict(x_test))
#        pr = metrics.precision_recall_curve(y_test, knn.predict_proba(x_test)[:,1])
#        results.append((recall, precision, auc, pr))
        results.append(precision_at(knn, target, test_ind, 100))
    return results 


def train_direct_random(target, trials, train_size=TRAIN_SIZE):
    I_shuffled = np.arange(N)
    np.random.shuffle(I_shuffled)
    for I_train_start in np.arange(N, step=train_size):
        I_test = I_shuffled[:I_train_start]+I_shuffled[I_train_start+train_size:] 
        I_train = I_shuffled[I_train_start:I_train_start+train_size]
        xtrain,ytrain = X[I_train], Y[I_train]==target
        scores  = eval_testset(target,I_test)

        # randomly chosen training set i.e. ~1/2 positive




def eval_interact_posttrain(t, train_size, test_size, repeats=50):
    results = []
    for i in range(repeats):
        m = test_target(t, train_size)
        train_hits = m.feedback[1].sum()
        m = test_target(t, test_size+train_size, m_init=m)
        total_hits = m.feedback[1].sum()
        results.append((train_hits, total_hits))
    return results
    
