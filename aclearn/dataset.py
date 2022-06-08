from matplotlib.transforms import Transform
from torchvision.datasets import MNIST,FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np


class AcLearnDataset():
    def __init__(self, dataset_path='demo_mnist', size_init_per_class=1):
        """
        """

        if (dataset_path=='demo_mnist') or (dataset_path=='demo_fmnist'): 
            train, test, self.nb_class, self.image_size = self.read_famous_dataset(dataset_path)
        else:
            train, test, self.nb_class, self.image_size = self.read_dataset(dataset_path)
        
        self.X_train, self.y_train = train
        self.X_test, self.y_test = test

        self.preprocessing()
        self.init_label_pool(size_init_per_class)



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
        
        X_train, y_train = next(iter(traindataloader))
        X_train, y_train = X_train.numpy(), y_train.numpy()
        X_test , y_test = next(iter(testdataloader))
        X_test , y_test = X_test.numpy() , y_test.numpy()

        nb_class = len(set(y_test))
        image_size = X_train.shape[2:]

        return (X_train, y_train), (X_test, y_test), nb_class, image_size
    

    def read_dataset(self, path):
        """
        """

        #return X_train, y_train, X_test, y_test, nb_class, image_size


    def preprocessing(self):
        """
        """
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.image_size[0], self.image_size[1])
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.image_size[0], self.image_size[1])
    
    

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