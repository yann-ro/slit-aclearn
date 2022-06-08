from torchvision.datasets import MNIST,FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np


class AcLearnDataset():
    def __init__(self, size_init_per_class=1, dataset_name='MNIST'):
        """
        """
        self.name = dataset_name
        self.read_data_demo()
        self.preprocessing()
        self.init_label_pool(size_init_per_class)



    def read_data_demo(self):
        """
        """
        train,test=None,None

        if self.name == 'MNIST':
            train = MNIST('.', train=True, download=True, transform=ToTensor())
            test  = MNIST('.', train=False,download=True, transform=ToTensor())
        elif self.name == 'FashionMNIST':
            train = FashionMNIST('.', train=True, download=True, transform=ToTensor())
            test = FashionMNIST('.', train=False, download=True, transform=ToTensor())
        
        traindataloader = DataLoader(train, shuffle=True, batch_size=60000)
        testdataloader  = DataLoader(test , shuffle=True, batch_size=10000)
        
        X_train, y_train = next(iter(traindataloader))
        self.X_train = X_train.detach().cpu().numpy()
        self.y_train = y_train.detach().cpu().numpy()
        
        X_test , y_test  = next(iter(testdataloader))
        self.X_test = X_test.detach().cpu().numpy()
        self.y_test = y_test.detach().cpu().numpy()

        self.nb_class = len(set(self.y_test))
        self.image_size = X_train.shape[2:]
    


    def preprocessing(self):
        """
        """
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.image_size[0], self.image_size[1])
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.image_size[0], self.image_size[1])
    
    

    def init_label_pool(self,nb_init_label_per_class):
        """
        """
        init_idx = np.array([],dtype=np.int)
        
        for i in range(self.nb_class):
            idx = np.random.choice(np.where(self.y_train==i)[0], size=nb_init_label_per_class, replace=False)
            init_idx = np.concatenate((init_idx, idx))

        self.X_init = self.X_train[init_idx]
        self.y_init = self.y_train[init_idx]
    
        self.X_pool = np.delete(self.X_train, init_idx, axis=0)
        self.y_pool = np.delete(self.y_train, init_idx, axis=0)