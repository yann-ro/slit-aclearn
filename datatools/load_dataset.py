from keras.datasets import mnist
import numpy as np


def load_random_mnist(nb_img=1):
    """_summary_

    Args:
        nb_img (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    (X_train, _), (_, _) = mnist.load_data()
    return X_train[np.random.randint(X_train.shape[0], size=nb_img)].squeeze()


def flat(X):
    """_summary_

    Args:
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    return X.reshape(X.shape[0], -1)
