import numpy as np

def sampling_random(X_pool,y_pool):
    nb_class = len(set(y_pool))
    if nb_class<3:
        index = np.random.choice(len(y_pool),1)
    else:
        index = np.random.choice(len(y_pool),nb_class)
    return X_pool[index].squeeze(), y_pool[index], np.delete(X_pool,index,axis=0), np.delete(y_pool,index)


def sampling_most_uncertain(clf,X_pool,y_pool):
    """
        extract most uncertain elements from X_pool
    """
    nb_class = clf.coef_.shape[0]
    if nb_class<3:
        index = np.argmin(np.abs(clf.decision_function(X_pool)))
    else:
        index = []
        for i in range(nb_class):
            index.append(np.argmin(np.abs(clf.decision_function(X_pool)[:,i])))
    return X_pool[index], y_pool[index], np.delete(X_pool,index,axis=0), np.delete(y_pool,index)