"""Factory for the classifiers used in the Iris comparative analysis."""

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def create_classifiers() -> dict:
    """Create the configured classifiers for the comparative analysis.

    Returns:
        dict: An insertion-ordered mapping from display names to classifier
            instances.
    """
    return {
        "C4.5 (Árbol de Decisión)": DecisionTreeClassifier(
            criterion="entropy",
            random_state=42,
        ),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=3),
        "Naive Bayes": GaussianNB(),
    }
