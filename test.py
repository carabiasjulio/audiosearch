from utils import *
from model import *
import datetime 

def test_target(t, n_iter):
    P = []
    S = []
    F = ([],[])

    m = SearchModel('test'+str(t)+'_'+str(datetime.datetime.now()))
    m.set_target_class(t)
    e, e_ind = m.get_random_class_sample()
    m.update_scores(1,1,1)

    def step(*mparams):
        m.update_scores(*mparams)
        proposals = m.get_proposals(5)
        #print "new proposal classes:", Y[[zip(*proposals)[1]]]
        return proposals

    def label_all(proposals):
        for s, ind, score in proposals:
            print score, Y[ind]
            m.add_feedback(Y[ind]==t, ind)


    for i in range(n_iter):
        F[0].append(len(m.get_feedback(0)))
        F[1].append(len(m.get_feedback(1)))
        proposals = step(max(3, m.get_trainset_size()/3), 1, 'euclidean')
        S.append( zip(*proposals)[2])
        P.append( Y[[zip(*proposals)[1]]])
        label_all(proposals)

    return P,S,F
#for i in [1865, 2260, 4038, 5469, 5472, 7679, 7682]:
#    m.feedback[0][i]=True
#for i in [5471, 5754, 5757, 5758, 6225, 6226]:
#    m.feedback[1][i]=True
