import numpy as np
import torch


def uniform(learner, X, query_size=1):
    query_idx = np.random.choice(range(len(X)), size=query_size, replace=False)
    return query_idx, X[query_idx]



def max_entropy(learner, X, query_size=1, T=100, max_sample=2000):
    
    if len(X)>max_sample: 
        random_subset = np.random.choice(range(len(X)), size=max_sample, replace=False)
    else:
        random_subset = np.arange(len(X))


    with torch.no_grad():
        outputs = np.stack([torch.softmax(learner.estimator.forward(X[random_subset], training=True),dim=-1).cpu().numpy()
                            for t in range(100)])
    pc = outputs.mean(axis=0)
    acquisition = (-pc*np.log(pc + 1e-10)).sum(axis=-1)
    idx = (-acquisition).argsort()[:query_size]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]



def bald(learner, X, query_size=1, T=100, max_sample=2000):
    
    if len(X)>max_sample: 
        random_subset = np.random.choice(range(len(X)), size=max_sample, replace=False)
    else:
        random_subset = np.arange(len(X))
    

    with torch.no_grad():
        outputs = np.stack([torch.softmax(learner.estimator.forward(X[random_subset], training=True),dim=-1).cpu().numpy()
                            for t in range(100)])
    pc = outputs.mean(axis=0)
    H   = (-pc*np.log(pc + 1e-10)).sum(axis=-1)
    E_H = - np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)  # [batch size]
    acquisition = H - E_H
    idx = (-acquisition).argsort()[:query_size]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]



def variation_ratio(learner, X, n_instances=1, T=100, max_sample=2000):
    
    if len(X)>max_sample: 
        random_subset = np.random.choice(range(len(X)), size=max_sample, replace=False)
    else:
        random_subset = np.arange(len(X))

    
    with torch.no_grad():
        outputs = np.stack([torch.softmax(learner.estimator.forward(X[random_subset], training=True), dim=-1).cpu().numpy()
                            for t in range(100)])  # p(y=c|x, w)
    pc = outputs.mean(axis=0)  #p(y=c|x, D_train) shape = (2000, 10)

    acquisition = 1 - np.amax(pc, axis=-1)  # TODO complete with variation ratio formula and pc
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]  