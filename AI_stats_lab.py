import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor
)


# -------------------------------------------------
# Question 1: Dataset generation and visualization
# -------------------------------------------------

def generate_clean_data(
    n_samples=500,
    noise=20,
    random_state=42
):
    """
    Generate a clean synthetic regression dataset.

    Return:
        X, y, true_coef

    Requirements:
    - n_samples = 500 by default
    - n_features = 1
    - n_informative = 1
    - noise = 20
    - coef = True
    - random_state = 42
    """
    X, y, true_coef = datasets.make_regression(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        noise=noise,
        coef=True,
        random_state=random_state
    )
    return X, y, true_coef

def add_outliers(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Add artificial outliers to the first n_outliers observations.

    Use:
        X[:n_outliers] = 10 + 0.75 * random_normal_values
        y[:n_outliers] = -15 + 20 * random_normal_values

    Return:
        X_out, y_out

    Important:
    Do not modify the original X and y directly.
    Make copies first.
    """
    rng = np.random.RandomState(random_state)
    X_out = X.copy()
    y_out = y.copy()
 
    X_out[:n_outliers] = 10 + 0.75 * rng.randn(n_outliers, X.shape[1])
    y_out[:n_outliers] = -15 + 20 * rng.randn(n_outliers)
 
    return X_out, y_out


def plot_dataset_with_outliers(
    X,
    y,
    n_outliers=25
):
    """
    Plot the dataset and highlight the first n_outliers observations.

    Return:
        matplotlib Figure object

    Requirements:
    - normal observations and artificial outliers should be visually different
    - include title
    - include x-label
    - include y-label
    - include legend
    """
    fig, ax = plt.subplots(figsize=(9, 6))
 
    ax.scatter(
        X[n_outliers:], y[n_outliers:],
        color="steelblue", alpha=0.6, edgecolors="white", linewidth=0.4,
        label="Normal observations"
    )
    ax.scatter(
        X[:n_outliers], y[:n_outliers],
        color="tomato", marker="X", s=80, edgecolors="darkred", linewidth=0.6,
        label=f"Artificial outliers (n={n_outliers})"
    )
 
    ax.set_title("Synthetic Regression Dataset with Outliers")
    ax.set_xlabel("Feature X")
    ax.set_ylabel("Target y")
    ax.legend()
    plt.tight_layout()
    return fig
 


# -------------------------------------------------
# Question 2: Fit regression models
# -------------------------------------------------

def fit_linear_regression(X, y):
    """
    Fit ordinary Linear Regression.

    Return:
        fitted coefficient as a float
    """
    model = LinearRegression()
    model.fit(X, y)
    return float(model.coef_[0])
 


def fit_huber_regression(X, y):
    """
    Fit Huber Regression.

    Return:
        fitted coefficient as a float
    """
    model = HuberRegressor()
    model.fit(X, y)
    return float(model.coef_[0])


def fit_ransac_regression(X, y, random_state=42):
    """
    Fit RANSAC Regression.

    Return:
        fitted coefficient as a float

    Hint:
    RANSAC stores the final linear model in estimator_.
    """
    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)
    return float(model.estimator_.coef_[0])


def fit_theilsen_regression(X, y, random_state=42):
    """
    Fit Theil-Sen Regression.

    Return:
        fitted coefficient as a float
    """
    model = TheilSenRegressor(random_state=random_state)
    model.fit(X, y)
    return float(model.coef_[0])


def coefficient_errors(coef_dict, true_coef):
    """
    Given a dictionary of coefficients and the true coefficient,
    return a dictionary of absolute coefficient errors.

    Example input:
        {
            "linear_regression": 8.7,
            "huber_regression": 37.5,
            "ransac_regression": 62.8,
            "theilsen_regression": 59.4
        }

    Return:
        {
            "linear_regression": abs(...),
            ...
        }
    """
    return {
        "linear_regression": abs(coef_dict["linear_regression"] - true_coef),
        "huber_regression": abs(coef_dict["huber_regression"] - true_coef),
        "ransac_regression": abs(coef_dict["ransac_regression"] - true_coef),
        "theilsen_regression": abs(coef_dict["theilsen_regression"] - true_coef)
    }


def best_robust_model(errors):
    """
    Return the name of the robust model with the smallest error.

    Only compare:
        huber_regression
        ransac_regression
        theilsen_regression

    Do not include ordinary linear_regression in this comparison.
    """
    return min(errors, key=errors.get)


def ransac_outlier_summary(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Fit RANSAC and return:

        total_outliers_detected, added_outliers_detected

    total_outliers_detected:
        total number of samples classified as outliers by RANSAC

    added_outliers_detected:
        number of artificial outliers among the first n_outliers
        that RANSAC classified as outliers
    """
    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)
    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask
    total_outliers_detected = int(np.sum(outlier_mask))
    added_outliers_detected = int(np.sum(outlier_mask[:n_outliers]))
    return total_outliers_detected, added_outliers_detected


# -------------------------------------------------
# Question 2: Visualization functions
# -------------------------------------------------

def plot_regression_fits(
    X,
    y,
    random_state=42
):
    """
    Plot fitted regression lines for:
    - Linear Regression
    - Huber Regression
    - RANSAC Regression
    - Theil-Sen Regression

    Return:
        matplotlib Figure object

    Requirements:
    - scatter plot of data
    - fitted line for each model
    - title
    - x-label
    - y-label
    - legend
    """
    # Fit all models
    lr   = LinearRegression().fit(X, y)
    hub  = HuberRegressor().fit(X, y)
    ran  = RANSACRegressor(random_state=random_state).fit(X, y)
    tsen = TheilSenRegressor(random_state=random_state).fit(X, y)
 
    X_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
 
    fig, ax = plt.subplots(figsize=(10, 6))
 
    ax.scatter(X, y, color="lightgray", alpha=0.5, edgecolors="gray",
               linewidth=0.3, label="Data", zorder=1)
 
    models = [
        (lr,   "Linear Regression",  "royalblue",  "-"),
        (hub,  "Huber Regression",   "darkorange", "--"),
        (ran,  "RANSAC Regression",  "green",      "-."),
        (tsen, "Theil-Sen",          "crimson",    ":"),
    ]
 
    for model, label, color, ls in models:
        ax.plot(X_line, model.predict(X_line),
                color=color, linestyle=ls, linewidth=2, label=label)
 
    ax.set_title("Comparison of Robust Regression Models")
    ax.set_xlabel("Feature X")
    ax.set_ylabel("Target y")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_ransac_inliers_outliers(
    X,
    y,
    random_state=42
):
    """
    Fit RANSAC and visualize inliers vs outliers.

    Return:
        matplotlib Figure object

    Requirements:
    - inliers and outliers should be visually different
    - title
    - x-label
    - y-label
    - legend
    """
    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)
    outlier_mask = ~model.inlier_mask_
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X[model.inlier_mask_], y[model.inlier_mask_], color="steelblue", alpha=0.6, edgecolors="white", linewidth=0.4, label="Inliers")
    ax.scatter(X[outlier_mask], y[outlier_mask], color="tomato", marker="X", s=80, edgecolors="darkred", linewidth=0.6, label="Outliers")
    ax.set_title("RANSAC Inliers vs Outliers")
    ax.set_xlabel("Feature X")
    ax.set_ylabel("Target y")
    ax.legend()
    plt.tight_layout()
    return fig