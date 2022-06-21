from aclearn.acquisition import uniform,max_entropy,bald,variation_ratio
from sklearn.metrics import ConfusionMatrixDisplay
from mlinsights.mlmodel import PredictableTSNE
from sklearn.metrics import confusion_matrix
from aclearn.dataset import AcLearnDataset
from sklearn.decomposition import PCA
from aclearn.estimator import CNN
from sklearn.manifold import TSNE
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt


from skorch import NeuralNetClassifier
from modAL.models import ActiveLearner
import numpy as np
import torch


class AcLearnModel():

    def __init__(self, query_strategy, dataset, model_id, oracle='computer', device='cpu'):
        """
        query_strategy (func):
        is_oracle (bool):
        """

        self.model_id = model_id

        print(f'$({self.model_id}) init AcLearn model')
        if query_strategy == 'Uniform': self.query_strategy = uniform
        elif query_strategy == 'Max_entropy': self.query_strategy = max_entropy
        elif query_strategy == 'Bald': self.query_strategy = bald
        elif query_strategy == 'Var_ratio': self.query_strategy = variation_ratio
        else : print('Not existing query_strategy')

        self.oracle = oracle
        self.dataset = dataset

        self.estimator = NeuralNetClassifier(CNN,
                                max_epochs=50,
                                batch_size=128,
                                lr=0.001,
                                optimizer=torch.optim.Adam,
                                criterion=torch.nn.CrossEntropyLoss,
                                train_split=None,
                                verbose=0,
                                device=device)

        self.acc_history = None
        self.max_accuracy = 0
        self.index_epoch = 0

        self.tsne = None
        self.pca = None
        self.emb_fig = None
        self.dataset.X_query = None
        self.dataset.y_query = None


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

        print(f'$({self.model_id}) init_training complete')



    def active_learning_procedure(self, n_queries=10, query_size=10, train_acc=False, progress_bar=None):
        """
        """
        
        if self.oracle=='computer':
            for i in range(n_queries):
                self.index_epoch += 1
                self.forward(query_size, train_acc)
                
                if train_acc:
                    print(f'\t(query {self.index_epoch}) Train acc: \t{self.acc_train_history[self.index_epoch]:0.4f}  |  Test acc: \t{self.acc_history[self.index_epoch]:0.4f}')
                else:
                    print(f'\t(query {self.index_epoch}) Test acc: \t{self.acc_history[ self.index_epoch]:0.4f}')
                
                progress_bar.progress(int((i+1)/n_queries*100))

        elif self.oracle=='human':
            print('MODE not implemented yet')
            self.index_epoch += 1



    def forward(self, query_size, train_acc):
        """
        """
        
        query_idx, query_instance = self.learner.query(self.dataset.X_pool, query_size)

        self.X_query = self.dataset.X_pool[query_idx]
        self.y_query = self.dataset.y_pool[query_idx]
        
        self.learner.teach(self.X_query, self.y_query)
        self.dataset.X_pool = np.delete(self.dataset.X_pool, query_idx, axis=0)
        self.dataset.y_pool = np.delete(self.dataset.y_pool, query_idx, axis=0)
        
        self.acc_history.append(self.learner.score(self.dataset.X_test, self.dataset.y_test))

        if train_acc: 
            self.acc_train_history.append(self.learner.score(self.dataset.X_train, self.dataset.y_train))
    


    def plot_confusion(self, labels=None, normalize=None, ax=None, cmap='viridis'):
        return ConfusionMatrixDisplay.from_estimator(self.learner.estimator, 
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



    def compute_tsne(self):
        """
        """
        X = self.learner.X_training.reshape(len(self.learner.X_training), -1)
        y = self.learner.y_training

        self.tsne = PredictableTSNE(transformer=TSNE(n_iter=1000, init='random', learning_rate='auto'))
        self.tsne.fit(X, y)
    
    

    def compute_pca(self):
        """
        """
        X = self.learner.X_training.reshape(len(self.learner.X_training), -1)
        y = self.learner.y_training

        self.pca = PCA().fit(X, y)

    
    def compute_emb_figure(self):
        """
        """

        X_train = self.learner.X_training.reshape(len(self.learner.X_training), -1)
        y_train = self.learner.y_training
        X_query = self.X_query.reshape(len(self.X_query), -1)
        y_query = self.y_query
        X_pool = self.dataset.X_pool.reshape(len(self.dataset.X_pool), -1)

        self.emb_fig = plot_results(X_train, 
                                    y_train, 
                                    X_query, 
                                    y_query, 
                                    X_pool, 
                                    tsne=self.tsne, 
                                    #pca=self.pca
                                    )




def confidence_ellipse(x2d, labels, ax, n_std=2.0, cm='none', **kwargs):
    """
    """
    
    classes = set(labels)
    colors = cm(np.linspace(0, 1, len(classes)))

    ellipses = []
    
    for c, color in zip(classes, colors):

        x, y = x2d[labels==c, 0], x2d[labels==c, 1]

        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            edgecolor='white',
            facecolor=color,
            linewidth=3,
            **kwargs)

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        ellipses.append(ellipse)

        ax.add_patch(ellipse)
    
    ax.legend(ellipses, classes)



def plot_results(X_train, y_train, X_selected, y_selected, X_pool, tsne=None, pca=None):
    """
    """
    
    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    
    if tsne:
        x2d_train_tsne = tsne.transform(X_train)
        x2d_selected_tsne = tsne.transform(X_selected)
        x2d_pool_tsne = tsne.transform(X_pool)
        
        print(x2d_train_tsne.shape)
        
        confidence_ellipse(x2d_train_tsne, y_train, ax[0], n_std=2, cm=cm.magma, alpha=0.4)
        
        ax[0].scatter(x2d_pool_tsne[:,0], x2d_pool_tsne[:,1], color='gray', alpha=0.05)
        ax[0].scatter(x2d_selected_tsne[:,0], x2d_selected_tsne[:,1], c=y_selected, cmap='magma')
        ax[0].set_title('t-SNE')

    if pca:
        x2d_train_pca = pca.transform(X_train)
        x2d_selected_pca = pca.transform(X_selected)
        x2d_pool_pca = pca.transform(X_pool)

        confidence_ellipse(x2d_train_pca, y_train, ax[1], n_std=2, cm=cm.magma, alpha=0.4)
        
        ax[1].scatter(x2d_pool_pca[:,0], x2d_pool_pca[:,1], color='gray', alpha=0.05)
        ax[1].scatter(x2d_selected_pca[:,0], x2d_selected_pca[:,1], c=y_selected, cmap='magma')
        ax[0].set_title('PCA')

    return fig