# Internal lib
from aclearn.acquisition import uniform,max_entropy,bald,variation_ratio
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from aclearn.dataset import AcLearnDataset
from aclearn.estimator import CNN
# External lib
from skorch import NeuralNetClassifier
from modAL.models import ActiveLearner
import numpy as np
import torch


class AcLearnModel():

    def __init__(self, query_strategy, dataset, is_oracle=False, device='cpu'):
        """
        query_strategy (func):
        is_oracle (bool):
        """

        print('$ init AcLearn model')
        if query_strategy == 'Uniform': self.query_strategy = uniform
        elif query_strategy == 'Max_entropy': self.query_strategy = max_entropy
        elif query_strategy == 'Bald': self.query_strategy = bald
        elif query_strategy == 'Var_ratio': self.query_strategy = variation_ratio
        else : print('Not existing query_strategy')
        
        self.is_oracle = is_oracle
        self.dataset = dataset
        self.device = device

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
        self.index_epoch = 0



    def evaluate_max(self):
        """
        """

        self.estimator.fit(self.dataset.X_train, self.dataset.y_train)
        self.max_accuracy = self.estimator.score(self.dataset.X_test, self.dataset.y_test)



    def init_training(self):
        """
        """

        self.learner = ActiveLearner(estimator=self.estimator,
                                X_training = self.dataset.X_init,
                                y_training = self.dataset.y_init,
                                query_strategy = self.query_strategy)

        self.acc_history = [self.learner.score(self.dataset.X_test, self.dataset.y_test)]
        self.acc_train_history = [self.learner.score(self.dataset.X_train, self.dataset.y_train)]

        print('$ init_training complete')



    def active_learning_procedure(self, n_queries=10, query_size=10, train_acc=False, progress_bar=None):
        """
        """
        
        for i in range(n_queries):
            self.index_epoch += 1
            self.forward(query_size, train_acc)
            
            if train_acc:
                print(f'\t(query {self.index_epoch}) Train acc: \t{self.acc_train_history[self.index_epoch]:0.4f}  |  Test acc: \t{self.acc_history[self.index_epoch]:0.4f}')
            else:
                print(f'\t(query {self.index_epoch}) Test acc: \t{self.acc_history[ self.index_epoch]:0.4f}')
            
            progress_bar.progress(int((i+1)/n_queries*100))



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
    

    def plot_confusion(self, labels=None, normalize=None, ax=None, cmap='Blues'):
        return plot_confusion_matrix(self.learner.estimator, 
                                     self.dataset.X_test, 
                                     self.dataset.y_test, 
                                     labels=labels, 
                                     normalize=normalize, 
                                     ax=ax, 
                                     cmap=cmap)


    def evaluate_confusion(self):
        """
        """
        y_pred = self.learner.predict(self.dataset.X_test)
        return confusion_matrix(self.dataset.y_test, y_pred, labels=None, sample_weight=None, normalize=None)


