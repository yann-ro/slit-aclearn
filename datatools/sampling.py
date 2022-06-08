import numpy as np

def sampling_random(X_pool,y_pool,query_size=10):
    nb_class = len(set(y_pool))
    if nb_class<3:
        index = np.random.choice(len(y_pool),query_size)
    else:
        index = np.random.choice(len(y_pool),query_size)
    return X_pool[index].squeeze(), y_pool[index], np.delete(X_pool,index,axis=0), np.delete(y_pool,index)


def sampling_most_uncertain(clf,X_pool,y_pool,query_size=10):
    """
        extract most uncertain elements from X_pool
    """
    nb_class = clf.coef_.shape[0]
    if nb_class<3:
        index = list(np.argsort(np.abs(clf.decision_function(X_pool)))[:int(query_size/nb_class)])
    else:
        index = []
        for i in range(nb_class):
            # get the n=query_size/nb_class most uncertain points
            index+=list(np.argsort(np.abs(clf.decision_function(X_pool)[:,i]))[:int(query_size/nb_class)]) 
    return X_pool[index], y_pool[index], np.delete(X_pool,index,axis=0), np.delete(y_pool,index)