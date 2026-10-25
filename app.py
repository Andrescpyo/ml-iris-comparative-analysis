"""Streamlit interface for the Iris classifier comparison."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.figure import Figure

from src.classifier_factory import create_classifiers
from src.data_loader import get_iris_data
from src.evaluator import evaluate_model


FEATURES_3D = ["sepal length", "sepal width", "petal length"]
PETAL_FEATURES = ["petal length", "petal width"]


@st.cache_data
def load_dataset() -> tuple:
    """Load the complete Iris feature matrix and target vector.

    Returns:
        tuple: Complete features and target values reconstructed from the
            deterministic train/test partition.
    """
    X_train, X_test, y_train, y_test = get_iris_data()
    features = pd.concat([X_train, X_test], ignore_index=True)
    targets = pd.Series(pd.concat(
        [pd.Series(y_train), pd.Series(y_test)],
        ignore_index=True,
    ), name="class")
    return features, targets


@st.cache_data
def evaluate_classifiers() -> pd.DataFrame:
    """Train and evaluate every configured classifier.

    Returns:
        pandas.DataFrame: Comparative metrics for all configured models.
    """
    X_train, X_test, y_train, y_test = get_iris_data()
    results = [
        evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        for name, model in create_classifiers().items()
    ]
    return pd.DataFrame(results)


def build_2d_plot(features: pd.DataFrame, targets: pd.Series) -> Figure:
    """Build a petal-length versus petal-width scatter plot.

    Args:
        features: Iris feature matrix containing the petal columns.
        targets: Iris class labels used to color each point.

    Returns:
        Figure: Matplotlib figure containing the two-dimensional plot.
    """
    plot_data = features[PETAL_FEATURES].copy()
    plot_data["class"] = targets.to_numpy()
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=plot_data,
        x="petal length",
        y="petal width",
        hue="class",
        palette="viridis",
        ax=axis,
    )
    axis.set_title("Largo del Pétalo vs Ancho del Pétalo")
    axis.set_xlabel("Largo del Pétalo (LP)")
    axis.set_ylabel("Ancho del Pétalo (AP)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def build_3d_plot(features: pd.DataFrame, targets: pd.Series) -> Figure:
    """Build a three-dimensional scatter plot using three Iris features.

    Args:
        features: Iris feature matrix containing the selected dimensions.
        targets: Iris class labels used to color each point.

    Returns:
        Figure: Matplotlib figure containing the three-dimensional plot.
    """
    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        features[FEATURES_3D[0]],
        features[FEATURES_3D[1]],
        features[FEATURES_3D[2]],
        c=pd.Categorical(targets).codes,
        cmap="viridis",
        alpha=0.8,
    )
    axis.set_title("Visualización 3D del Dataset Iris")
    axis.set_xlabel("Largo del Sépalo")
    axis.set_ylabel("Ancho del Sépalo")
    axis.set_zlabel("Largo del Pétalo")
    figure.colorbar(scatter, ax=axis, pad=0.1, label="Clase")
    figure.tight_layout()
    return figure


def main() -> None:
    """Render the project description, metrics table, and visualizations."""
    st.set_page_config(
        page_title="Comparativa de Clasificadores Iris",
        page_icon="🌸",
        layout="wide",
    )
    st.title("Comparativa de algoritmos de clasificación con Iris")
    st.write(
        "Esta aplicación entrena y compara un árbol de decisión C4.5, "
        "K-Nearest Neighbors y Naive Bayes sobre el dataset Iris."
    )

    metrics = evaluate_classifiers()
    st.subheader("Resultados de los modelos")
    st.dataframe(
        metrics.rename(
            columns={
                "name": "Nombre",
                "accuracy": "Precisión",
                "training_time": "Tiempo de Entrenamiento (s)",
                "prediction_time": "Tiempo de Predicción (s)",
            }
        ).style.format({
            "Precisión": "{:.4f}",
            "Tiempo de Entrenamiento (s)": "{:.6f}",
            "Tiempo de Predicción (s)": "{:.6f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    features, targets = load_dataset()
    st.subheader("Visualización 2D: LP vs AP")
    st.pyplot(build_2d_plot(features, targets))
    plt.close("all")

    st.subheader("Visualización 3D")
    st.pyplot(build_3d_plot(features, targets))
    plt.close("all")


if __name__ == "__main__":
    main()
