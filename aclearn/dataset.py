from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST,FashionMNIST
from matplotlib.transforms import Transform
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np


class AcLearnDataset():
    def __init__(self, data_path, label_path=None, data_path_unlab=None, size_init_per_class=1):
        """
        """

        if (data_path=='demo_mnist') or (data_path=='demo_fmnist'): 
            self.read_famous_dataset(data_path)
        else:
            self.read_dataset(data_path, label_path, data_path_unlab, extension='npy')
        
        if not data_path_unlab:
            self.init_label_pool(size_init_per_class)
        
        self.preprocessing()




    def read_famous_dataset(self, demo_name):
        """
        """

        if demo_name == 'demo_mnist':
            train = MNIST('.', train=True, download=True, transform=ToTensor())
            test  = MNIST('.', train=False, download=True, transform=ToTensor())

        elif demo_name == 'demo_fmnist':
            train = FashionMNIST('.', train=True, download=True, transform=ToTensor())
            test = FashionMNIST('.', train=False, download=True, transform=ToTensor())
        
        traindataloader = DataLoader(train, shuffle=True, batch_size=60000)
        testdataloader  = DataLoader(test , shuffle=True, batch_size=10000)
        
        self.X_train, self.y_train = next(iter(traindataloader))
        self.X_train, self.y_train = self.X_train.numpy(), self.y_train.numpy()
        self.X_test , self.y_test = next(iter(testdataloader))
        self.X_test , self.y_test = self.X_test.numpy() , self.y_test.numpy()

        self.nb_class = len(set(self.y_test))
        self.image_size = self.X_train.shape[-2:]
    


    def read_dataset(self, data_path, label_path, data_path_unlab, extension='npy'):
        """
        """

        if extension == 'npy':
            X = np.load(data_path).reshape(-1, 28,28)
            y = np.array(np.load(label_path).squeeze(), dtype=int)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

            if data_path_unlab:
                self.X_pool = np.load(data_path_unlab)
                self.y_pool = None

            self.nb_class = len(set(y))
            self.image_size = X.shape[-2:]

        else :
            print('ERROR: File extension not supported')
        



    def preprocessing(self):
        """
        """
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.image_size[0], self.image_size[1])
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.image_size[0], self.image_size[1])

        self.X_pool = self.X_pool.reshape(self.X_pool.shape[0], 1, self.image_size[0], self.image_size[1])

    

    def init_label_pool(self,nb_init_label_per_class):
        """
        """
        init_idx = np.array([], dtype=np.int)
        
        for i in range(self.nb_class):
            idx = np.random.choice(np.where(self.y_train==i)[0], size=nb_init_label_per_class, replace=False)
            init_idx = np.concatenate((init_idx, idx))

        self.X_init = self.X_train[init_idx]
        self.y_init = self.y_train[init_idx]
    
        self.X_pool = np.delete(self.X_train, init_idx, axis=0)
        self.y_pool = np.delete(self.y_train, init_idx, axis=0)