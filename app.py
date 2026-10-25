"""Streamlit interface for the interactive Iris classifier analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.figure import Figure
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

from src.classifier_factory import create_classifiers
from src.data_loader import get_iris_data
from src.evaluator import evaluate_model


FEATURE_NAMES = [
    "sepal length",
    "sepal width",
    "petal length",
    "petal width",
]
PETAL_FEATURES = ["petal length", "petal width"]
FEATURE_LABELS = {
    "sepal length": "Largo del Sépalo",
    "sepal width": "Ancho del Sépalo",
    "petal length": "Largo del Pétalo",
    "petal width": "Ancho del Pétalo",
}
PRESETS = {
    "Ejemplo Setosa": {
        "sepal length": 5.0,
        "sepal width": 3.4,
        "petal length": 1.5,
        "petal width": 0.2,
    },
    "Ejemplo Versicolor": {
        "sepal length": 5.9,
        "sepal width": 2.8,
        "petal length": 4.3,
        "petal width": 1.3,
    },
    "Ejemplo Virginica": {
        "sepal length": 6.5,
        "sepal width": 3.0,
        "petal length": 5.5,
        "petal width": 2.0,
    },
}


@st.cache_data
def load_dataset() -> tuple:
    """Load the complete dataset and its reproducible train/test partition.

    Returns:
        tuple: Complete features, complete targets, and the train/test split.
    """
    X_train, X_test, y_train, y_test = get_iris_data()
    features = pd.concat([X_train, X_test], ignore_index=True)[FEATURE_NAMES]
    targets = pd.Series(
        pd.concat(
            [pd.Series(y_train), pd.Series(y_test)],
            ignore_index=True,
        ),
        name="class",
    )
    return features, targets, (X_train, X_test, y_train, y_test)


def fit_models(
    knn_neighbors: int = 3,
    tree_max_depth: int | None = None,
) -> dict:
    """Fit the configured classifiers on the complete feature set.

    Args:
        knn_neighbors: Number of neighbors used by KNN.
        tree_max_depth: Maximum depth used by the decision tree.

    Returns:
        dict: Fitted classifiers keyed by their display names.
    """
    X_train, _, y_train, _ = load_dataset()[2]
    models = create_classifiers(knn_neighbors, tree_max_depth)
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_models(
    knn_neighbors: int = 3,
    tree_max_depth: int | None = None,
) -> pd.DataFrame:
    """Evaluate every configured model with the standard train/test split.

    Args:
        knn_neighbors: Number of neighbors used by KNN.
        tree_max_depth: Maximum depth used by the decision tree.

    Returns:
        pandas.DataFrame: Accuracy and execution times for each model.
    """
    X_train, X_test, y_train, y_test = load_dataset()[2]
    results = [
        evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        for name, model in create_classifiers(
            knn_neighbors,
            tree_max_depth,
        ).items()
    ]
    return pd.DataFrame(results)


def build_exploration_2d(features: pd.DataFrame, targets: pd.Series) -> Figure:
    """Build the pure LP versus AP dataset exploration plot.

    Args:
        features: Complete Iris feature matrix.
        targets: Complete Iris target vector.

    Returns:
        Figure: Scatter plot without model regions or predictions.
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
    axis.set_title("Exploración del Dataset: Largo vs Ancho del Pétalo")
    axis.set_xlabel("Largo del Pétalo (LP)")
    axis.set_ylabel("Ancho del Pétalo (AP)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def build_exploration_3d(
    features: pd.DataFrame,
    targets: pd.Series,
) -> Figure:
    """Build the pure three-dimensional dataset exploration plot.

    Args:
        features: Complete Iris feature matrix.
        targets: Complete Iris target vector.

    Returns:
        Figure: Three-dimensional scatter plot colored by class.
    """
    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        features["sepal length"],
        features["sepal width"],
        features["petal length"],
        c=pd.Categorical(targets).codes,
        cmap="viridis",
        alpha=0.8,
    )
    axis.set_title("Exploración 3D del Dataset Iris")
    axis.set_xlabel("Largo del Sépalo")
    axis.set_ylabel("Ancho del Sépalo")
    axis.set_zlabel("Largo del Pétalo")
    figure.colorbar(scatter, ax=axis, pad=0.1, label="Clase")
    figure.tight_layout()
    return figure


def build_decision_boundary(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Figure:
    """Build LP/AP decision regions and highlight test-set errors.

    Args:
        model: Fitted full-feature classifier whose algorithm is visualized.
        X_train: Training feature matrix.
        X_test: Test feature matrix plotted over the regions.
        y_train: Training targets used to fit the two-dimensional projection.
        y_test: Test targets used to identify incorrect predictions.

    Returns:
        Figure: Decision-region plot with test observations and errors.
    """
    boundary_model = clone(model)
    boundary_model.fit(X_train[PETAL_FEATURES], y_train)
    x_values = X_train["petal length"]
    y_values = X_train["petal width"]
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_values.min() - 0.2, 8.0, 250),
        np.linspace(y_values.min() - 0.2, y_values.max() + 0.2, 250),
    )
    grid = pd.DataFrame({
        "petal length": x_grid.ravel(),
        "petal width": y_grid.ravel(),
    })
    grid_predictions = boundary_model.predict(grid)
    classes = list(boundary_model.classes_)
    class_codes = {label: code for code, label in enumerate(classes)}
    regions = np.array([class_codes[label] for label in grid_predictions])
    test_predictions = boundary_model.predict(X_test[PETAL_FEATURES])
    errors = np.asarray(y_test) != np.asarray(test_predictions)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.contourf(
        x_grid,
        y_grid,
        regions.reshape(x_grid.shape),
        levels=np.arange(len(classes) + 1) - 0.5,
        alpha=0.3,
        cmap="viridis",
    )
    # La malla fija sépalos implícitamente al excluirlos: así la geometría
    # muestra la separación que el algoritmo aprende específicamente en LP/AP.
    sns.scatterplot(
        x=X_test["petal length"],
        y=X_test["petal width"],
        hue=y_test,
        palette="viridis",
        edgecolor="black",
        ax=axis,
    )
    if errors.any():
        axis.scatter(
            X_test.loc[errors, "petal length"],
            X_test.loc[errors, "petal width"],
            marker="x",
            c="red",
            s=100,
            linewidths=2.5,
            label="Predicción incorrecta",
        )
    axis.set_title("Frontera de Decisión: LP vs AP (solo conjunto de prueba)")
    axis.set_xlabel("Largo del Pétalo (LP)")
    axis.set_ylabel("Ancho del Pétalo (AP)")
    axis.set_xlim(x_values.min() - 0.2, 8.0)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    return figure


def build_confusion_matrix(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Figure:
    """Build a confusion-matrix heatmap for one fitted model.

    Args:
        model: Fitted classifier selected for the individual analysis.
        X_test: Test feature matrix.
        y_test: Test target vector.

    Returns:
        Figure: Seaborn heatmap of real versus predicted classes.
    """
    predictions = model.predict(X_test)
    labels = list(model.classes_)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_title("Matriz de Confusión")
    axis.set_xlabel("Clase Predicha")
    axis.set_ylabel("Clase Real")
    figure.tight_layout()
    return figure


def render_simulator(features: pd.DataFrame) -> None:
    """Render preset controls, flower inputs, and submitted predictions.

    Args:
        features: Complete feature matrix used to define valid slider ranges.
    """
    st.subheader("Simulador de una flor Iris")
    preset_columns = st.columns(len(PRESETS))
    for column, (preset_name, preset_values) in zip(
        preset_columns,
        PRESETS.items(),
    ):
        with column:
            if st.button(preset_name, use_container_width=True):
                for feature, value in preset_values.items():
                    st.session_state[f"simulator_{feature}"] = value

    for feature in FEATURE_NAMES:
        state_key = f"simulator_{feature}"
        if state_key not in st.session_state:
            st.session_state[state_key] = float(features[feature].median())

    with st.form("iris_simulator_form"):
        values = {}
        columns = st.columns(4)
        for column, feature in zip(columns, FEATURE_NAMES):
            with column:
                values[feature] = st.slider(
                    FEATURE_LABELS[feature],
                    min_value=float(features[feature].min()),
                    max_value=float(features[feature].max()),
                    step=0.1,
                    key=f"simulator_{feature}",
                )
        submitted = st.form_submit_button("Simular", type="primary")

    if not submitted:
        return

    sample = pd.DataFrame([values], columns=FEATURE_NAMES)
    models = fit_models()
    st.subheader("Predicciones")
    result_columns = st.columns(len(models))
    for column, (name, model) in zip(result_columns, models.items()):
        prediction = str(model.predict(sample)[0])
        with column:
            if "setosa" in prediction.lower():
                st.success(f"🌸 {name}: {prediction}")
            elif "versicolor" in prediction.lower():
                st.info(f"🌼 {name}: {prediction}")
            else:
                st.warning(f"🌺 {name}: {prediction}")


def render_general_view(
    features: pd.DataFrame,
    targets: pd.Series,
) -> None:
    """Render global metrics and model-independent dataset exploration.

    Args:
        features: Complete Iris feature matrix.
        targets: Complete Iris target vector.
    """
    st.header("Vista General")
    st.write(
        "Comparación global de los clasificadores y exploración visual pura "
        "del dataset, sin regiones calculadas por modelos."
    )
    st.subheader("Tabla comparativa global")
    metrics = evaluate_models()
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
    st.subheader("Exploración 2D: LP vs AP")
    st.pyplot(build_exploration_2d(features, targets), use_container_width=True)
    plt.close("all")
    st.subheader("Exploración 3D")
    st.pyplot(build_exploration_3d(features, targets), use_container_width=True)
    plt.close("all")


def render_individual_view(
    split: tuple,
) -> None:
    """Render one model's hyperparameters, confusion matrix, and boundary.

    Args:
        split: Train/test features and targets from the cached dataset.
    """
    X_train, X_test, y_train, y_test = split
    st.header("Análisis Individual")
    selected_name = st.selectbox(
        "Algoritmo",
        [
            "C4.5 (Árbol de Decisión)",
            "K-Nearest Neighbors (KNN)",
            "Naive Bayes",
        ],
    )
    knn_neighbors = 3
    tree_max_depth = None
    if selected_name == "K-Nearest Neighbors (KNN)":
        knn_neighbors = st.slider("K de KNN", 1, 15, 3)
    elif selected_name == "C4.5 (Árbol de Decisión)":
        tree_max_depth = st.slider(
            "Profundidad máxima del Árbol de Decisión",
            1,
            10,
            5,
        )

    model = create_classifiers(knn_neighbors, tree_max_depth)[selected_name]
    model.fit(X_train, y_train)
    st.subheader("Matriz de Confusión")
    st.pyplot(
        build_confusion_matrix(model, X_test, y_test),
        use_container_width=True,
    )
    plt.close("all")
    st.subheader("Frontera de Decisión")
    st.pyplot(
        build_decision_boundary(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        ),
        use_container_width=True,
    )
    plt.close("all")


def main() -> None:
    """Route the application to exactly one of its three views."""
    st.set_page_config(
        page_title="Comparativa de Clasificadores Iris",
        page_icon="🌸",
        layout="wide",
    )
    st.title("Experimentación interactiva con clasificadores Iris")
    st.sidebar.header("Navegación")
    selected_view = st.sidebar.radio(
        "Selecciona una vista",
        ["Vista General", "Análisis Individual", "Simulador"],
    )
    features, targets, split = load_dataset()

    if selected_view == "Vista General":
        render_general_view(features, targets)
    elif selected_view == "Análisis Individual":
        render_individual_view(split)
    else:
        render_simulator(features)


if __name__ == "__main__":
    main()
