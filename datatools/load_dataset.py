from keras.datasets import mnist
import numpy as np

def load_random_mnist(nb_img=1):
    (X_train,_), (_,_) = mnist.load_data()
    return X_train[np.random.randint(X_train.shape[0],size=nb_img)].squeeze()


def flat(X):
    return X.reshape(X.shape[0],-1)