import matplotlib.pyplot as plt
import numpy as np


def plot_iter_active(clf, X_pool, X_train, y_train, X_mu, ground_truth):
    """_summary_

    Args:
        clf (_type_): _description_
        X_pool (_type_): _description_
        X_train (_type_): _description_
        y_train (_type_): _description_
        X_mu (_type_): _description_
        ground_truth (_type_): _description_
    """

    plt.scatter(X_pool[:, 0], X_pool[:, 1], c="grey")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=100)
    plt.scatter(X_mu[0], X_mu[1], c="red", marker="*", s=150)

    x_min, x_max = ground_truth["X_min"], ground_truth["X_max"]
    step_x = (x_max - x_min) / 100

    x1_sample = np.array([x_min + i * step_x for i in range(100)])
    x2_sample = (-clf.coef_[0, 0] * x1_sample - clf.intercept_) / clf.coef_[0, 1]
    x2_true = (
        -ground_truth["alpha1"] * x1_sample - ground_truth["beta1"]
    ) / ground_truth["alpha2"]

    plt.plot(x1_sample, x2_sample, "--", c="orange", alpha=1)
    plt.plot(x1_sample, x2_true, "--", c="r", alpha=0.2)
