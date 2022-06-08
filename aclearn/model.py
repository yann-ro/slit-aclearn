# Internal lib
from aclearn.acquisition import uniform,max_entropy,bald,variation_ratio
from aclearn.dataset import AcLearnDataset
from aclearn.estimator import CNN
# External lib
from skorch import NeuralNetClassifier
from modAL.models import ActiveLearner
import numpy as np
import torch


class AcLearnModel():
    """
    """


    def __init__(self, query_strategy, dataset, logger, is_oracle=False):
        """
        query_strategy (func):
        is_oracle (bool):
        """
        if query_strategy == 'uniform': self.query_strategy = uniform
        elif query_strategy == 'max_entropy': self.query_strategy = max_entropy
        elif query_strategy == 'bald': self.query_strategy = bald
        elif query_strategy == 'variation_ratio': self.query_strategy = variation_ratio
        else : print('Not existing query_strategy')
        
        self.logger = logger
        self.is_oracle = is_oracle
        self.dataset = dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.estimator = NeuralNetClassifier(CNN,
                                max_epochs=50,
                                batch_size=128,
                                lr=0.001,
                                optimizer=torch.optim.Adam,
                                criterion=torch.nn.CrossEntropyLoss,
                                train_split=None,
                                verbose=0,
                                device=self.device)

        self.acc_history = None
        self.max_accuracy = 0



    def evaluate_max(self):
        """
        """
        self.estimator.fit(self.dataset.X_train, self.dataset.y_train)
        self.max_accuracy = self.estimator.score(self.dataset.X_test, self.dataset.y_test)



    def active_learning_procedure(self, n_queries=10, query_size=10,train_acc=False):
        """
        """
        self.learner = ActiveLearner(estimator=self.estimator,
                                X_training = self.dataset.X_init,
                                y_training = self.dataset.y_init,
                                query_strategy = self.query_strategy)

        self.acc_history = [self.learner.score(self.dataset.X_test, self.dataset.y_test)]
        if train_acc: 
            self.acc_train_history = [self.learner.score(self.dataset.X_train, self.dataset.y_train)]

        for index in range(n_queries):
            self.forward(query_size,train_acc)
            
            if train_acc:
                print(f'(query {index+1}) Train acc: \t{self.acc_train_history[index]:0.4f}  |  Test acc: \t{self.acc_history[index]:0.4f}')
            else:
                print(f'(query {index+1}) Test acc: \t{self.acc_history[index]:0.4f}')



    def forward(self,query_size,train_acc):
        """
        """
        query_idx, query_instance = self.learner.query(self.dataset.X_pool, query_size)
        
        self.learner.teach(self.dataset.X_pool[query_idx], self.dataset.y_pool[query_idx])
        self.dataset.X_pool = np.delete(self.dataset.X_pool, query_idx, axis=0)
        self.dataset.y_pool = np.delete(self.dataset.y_pool, query_idx, axis=0)
        
        self.acc_history.append(self.learner.score(self.dataset.X_test, self.dataset.y_test))
        if train_acc: 
            self.acc_train_history.append(self.learner.score(self.dataset.X_train, self.dataset.y_train))
    

