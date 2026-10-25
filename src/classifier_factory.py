"""Factory for the classifiers used in the Iris comparative analysis."""

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def create_classifiers(
    knn_neighbors: int = 3,
    tree_max_depth: int | None = None,
) -> dict:
    """Create the configured classifiers for the comparative analysis.

    Args:
        knn_neighbors: Number of neighbors used by KNN.
        tree_max_depth: Maximum depth of the decision tree.

    Returns:
        dict: An insertion-ordered mapping from display names to classifier
            instances.
    """
    return {
        "C4.5 (Árbol de Decisión)": DecisionTreeClassifier(
            criterion="entropy",
            random_state=42,
            max_depth=tree_max_depth,
        ),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(
            n_neighbors=knn_neighbors,
        ),
        "Naive Bayes": GaussianNB(),
    }
