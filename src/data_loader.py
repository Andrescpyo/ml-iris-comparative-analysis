"""Load and partition the Iris dataset provided by the UCI repository."""

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


def get_iris_data(test_size=0.3, random_state=42):
    """Fetch the Iris dataset and split it into training and test sets.

    Args:
        test_size: Fraction of samples assigned to the test set.
        random_state: Seed used to make the partition reproducible.

    Returns:
        tuple: Training features, test features, training targets, and test
            targets, in the order returned by ``train_test_split``.
    """
    iris = fetch_ucirepo(id=53) 
    X = iris.data.features 

    # Flattening prevents scikit-learn from treating one-column targets as
    # two-dimensional data and avoids estimator format warnings.
    y = iris.data.targets.values.ravel() 

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )