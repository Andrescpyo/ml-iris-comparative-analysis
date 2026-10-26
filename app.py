"""Streamlit interface for the interactive Iris classifier analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.figure import Figure
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance

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


def build_sepal_exploration(
    features: pd.DataFrame,
    targets: pd.Series,
) -> Figure:
    """Build the pure SL versus SW dataset exploration plot.

    Args:
        features: Complete Iris feature matrix.
        targets: Complete Iris target vector.

    Returns:
        Figure: Sepal scatter plot without model regions or predictions.
    """
    plot_data = features[["sepal length", "sepal width"]].copy()
    plot_data["class"] = targets.to_numpy()
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=plot_data,
        x="sepal length",
        y="sepal width",
        hue="class",
        palette="viridis",
        ax=axis,
    )
    axis.set_title("Exploración del Dataset: Largo vs Ancho del Sépalo")
    axis.set_xlabel("Largo del Sépalo (LS)")
    axis.set_ylabel("Ancho del Sépalo (AS)")
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
    feature_pair: list[str] | tuple[str, str] = PETAL_FEATURES,
) -> Figure:
    """Build two-dimensional decision regions and highlight test errors.

    Args:
        model: Fitted full-feature classifier whose algorithm is visualized.
        X_train: Training feature matrix.
        X_test: Test feature matrix plotted over the regions.
        y_train: Training targets used to fit the two-dimensional projection.
        y_test: Test targets used to identify incorrect predictions.
        feature_pair: Two feature names used as the projected plane.

    Returns:
        Figure: Decision-region plot with test observations and errors.
    """
    feature_pair = list(feature_pair)
    boundary_model = clone(model)
    boundary_model.fit(X_train[feature_pair], y_train)
    x_values = X_train[feature_pair[0]]
    y_values = X_train[feature_pair[1]]
    x_margin = max((x_values.max() - x_values.min()) * 0.08, 0.2)
    y_margin = max((y_values.max() - y_values.min()) * 0.08, 0.2)
    x_max = max(x_values.max() + x_margin, 8.0) if feature_pair == PETAL_FEATURES else x_values.max() + x_margin
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_values.min() - x_margin, x_max, 250),
        np.linspace(y_values.min() - y_margin, y_values.max() + y_margin, 250),
    )
    grid = pd.DataFrame({
        feature_pair[0]: x_grid.ravel(),
        feature_pair[1]: y_grid.ravel(),
    })
    grid_predictions = boundary_model.predict(grid)
    classes = list(boundary_model.classes_)
    class_codes = {label: code for code, label in enumerate(classes)}
    regions = np.array([class_codes[label] for label in grid_predictions])
    test_predictions = boundary_model.predict(X_test[feature_pair])
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
    # La proyección reentrena el algoritmo con solo dos variables para que
    # cada región corresponda exactamente al plano visualizado.
    sns.scatterplot(
        x=X_test[feature_pair[0]],
        y=X_test[feature_pair[1]],
        hue=y_test,
        palette="viridis",
        edgecolor="black",
        ax=axis,
    )
    if errors.any():
        axis.scatter(
            X_test.loc[errors, feature_pair[0]],
            X_test.loc[errors, feature_pair[1]],
            marker="x",
            c="red",
            s=100,
            linewidths=2.5,
            label="Predicción incorrecta",
        )
    axis.set_title(
        f"Frontera de Decisión: {FEATURE_LABELS[feature_pair[0]]} "
        f"vs {FEATURE_LABELS[feature_pair[1]]}"
    )
    axis.set_xlabel(FEATURE_LABELS[feature_pair[0]])
    axis.set_ylabel(FEATURE_LABELS[feature_pair[1]])
    axis.set_xlim(x_values.min() - x_margin, x_max)
    axis.set_ylim(y_values.min() - y_margin, y_values.max() + y_margin)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    return figure


def build_simulator_boundaries(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample: pd.DataFrame,
) -> Figure:
    """Build one petal boundary and virtual-flower point per classifier.

    Args:
        models: Fitted classifiers keyed by display name.
        X_train: Training features used to fit projected classifiers.
        y_train: Training targets used to calculate the regions.
        sample: One virtual flower with all four feature values.

    Returns:
        Figure: Horizontal three-panel figure with model regions and the
            virtual flower highlighted in every panel.
    """
    figure, axes = plt.subplots(
        1,
        len(models),
        figsize=(18, 5),
        squeeze=False,
    )
    x_values = X_train[PETAL_FEATURES[0]]
    y_values = X_train[PETAL_FEATURES[1]]
    x_margin = max((x_values.max() - x_values.min()) * 0.08, 0.2)
    y_margin = max((y_values.max() - y_values.min()) * 0.08, 0.2)
    x_max = max(x_values.max() + x_margin, 8.0)
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_values.min() - x_margin, x_max, 250),
        np.linspace(y_values.min() - y_margin, y_values.max() + y_margin, 250),
    )
    grid = pd.DataFrame({
        PETAL_FEATURES[0]: x_grid.ravel(),
        PETAL_FEATURES[1]: y_grid.ravel(),
    })

    for axis, (name, model) in zip(axes[0], models.items()):
        projected_model = clone(model)
        projected_model.fit(X_train[PETAL_FEATURES], y_train)
        predictions = projected_model.predict(grid)
        classes = list(projected_model.classes_)
        class_codes = {label: code for code, label in enumerate(classes)}
        regions = np.array([class_codes[label] for label in predictions])
        axis.contourf(
            x_grid,
            y_grid,
            regions.reshape(x_grid.shape),
            levels=np.arange(len(classes) + 1) - 0.5,
            alpha=0.35,
            cmap="viridis",
        )
        axis.scatter(
            sample["petal length"].iloc[0],
            sample["petal width"].iloc[0],
            marker="*",
            s=420,
            c="yellow",
            edgecolors="black",
            linewidths=1.8,
            zorder=3,
            label="Lirio virtual",
        )
        axis.set_title(name)
        axis.set_xlabel("Largo del Pétalo (LP)")
        axis.set_ylabel("Ancho del Pétalo (AP)")
        axis.set_xlim(x_values.min() - x_margin, x_max)
        axis.set_ylim(y_values.min() - y_margin, y_values.max() + y_margin)
        axis.grid(alpha=0.2)
        axis.legend(loc="best")

    figure.suptitle("Ubicación del lirio virtual en cada frontera de decisión")
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


def build_feature_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Figure:
    """Build feature-importance bars for any fitted classifier.

    Args:
        model: Fitted classifier trained on the four Iris features.
        X_test: Test feature matrix used for model-agnostic importance.
        y_test: Test target vector used to measure accuracy.

    Returns:
        Figure: Horizontal bar chart ordered by importance values.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_label = "Reducción de entropía normalizada"
    else:
        permutation = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="accuracy",
            n_repeats=20,
            random_state=42,
        )
        importances = permutation.importances_mean
        importance_label = "Disminución media de precisión"

    importance_data = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance": importances,
        }
    ).sort_values("importance", ascending=True)
    figure, axis = plt.subplots(figsize=(7, 3.5))
    sns.barplot(
        data=importance_data,
        x="importance",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
        ax=axis,
    )
    axis.set_title("Importancia de características")
    axis.set_xlabel(importance_label)
    axis.set_ylabel("Característica")
    minimum = importance_data["importance"].min()
    maximum = importance_data["importance"].max()
    margin = max((maximum - minimum) * 0.15, 0.05)
    axis.set_xlim(minimum - margin, maximum + margin)
    axis.axvline(0, color="black", linewidth=0.8)
    figure.tight_layout()
    return figure


def render_simulator(features: pd.DataFrame) -> None:
    """Render preset controls, flower inputs, and submitted predictions.

    Args:
        features: Complete feature matrix used to define valid slider ranges.
    """
    st.subheader("Simulador de una flor Iris")
    st.markdown(
        "**Inferencia de Datos Desconocidos:** Este simulador representa el "
        "uso del conocimiento generado por el sistema. Ajusta las 4 "
        "características biológicas para ver en tiempo real cómo cada modelo "
        "clasifica un ejemplo de naturaleza completamente desconocida "
        "basándose en las regiones de decisión aprendidas."
    )
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
    st.subheader("Ubicación geométrica del lirio virtual")
    X_train, _, y_train, _ = load_dataset()[2]
    st.pyplot(
        build_simulator_boundaries(models, X_train, y_train, sample),
        use_container_width=True,
    )
    plt.close("all")


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
    st.markdown(
        "**Introducción al Proyecto**\n"
        "Bienvenido a la plataforma de experimentación interactiva para el "
        "análisis de clasificadores de Aprendizaje Automático. Este proyecto "
        "de ingeniería, desarrollado para la Universidad Distrital, tiene "
        "como objetivo central entrenar, evaluar y comparar empíricamente el "
        "rendimiento de tres algoritmos fundamentales (Árboles de Decisión "
        "C4.5, K-Nearest Neighbors y Naive Bayes) aplicados al clásico "
        "conjunto de datos Iris. A través de este panel, se contrastarán "
        "las métricas de precisión y tiempos computacionales con el "
        "comportamiento geométrico de las fronteras de decisión, permitiendo "
        "una comprensión profunda de cómo cada modelo sintetiza el "
        "conocimiento para clasificar nuevas instancias."
    )
    st.markdown(
        "**Muestra de Datos Crudos:** A continuación se presenta un extracto "
        "de las instancias sin procesar. Cada fila representa una flor "
        "caracterizada por cuatro mediciones continuas y su etiqueta "
        "biológica, formando la base empírica sobre la cual los algoritmos "
        "sintetizarán el conocimiento."
    )
    raw_data = features.copy()
    raw_data["class"] = targets.to_numpy()
    st.dataframe(raw_data.head(10), use_container_width=True, hide_index=True)
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
    st.subheader("Exploración 2D")
    petal_column, sepal_column = st.columns(2)
    with petal_column:
        st.pyplot(
            build_exploration_2d(features, targets),
            use_container_width=True,
        )
        plt.close("all")
        st.caption(
            "Este plano bidimensional demuestra una alta separabilidad "
            "lineal. La clase Setosa se aísla completamente, lo que "
            "facilitará su clasificación matemática."
        )
    with sepal_column:
        st.pyplot(
            build_sepal_exploration(features, targets),
            use_container_width=True,
        )
        plt.close("all")
        st.caption(
            "A diferencia de los pétalos, los sépalos presentan una densa "
            "superposición espacial entre Versicolor y Virginica, "
            "convirtiéndose en variables ruidosas que inducen a la "
            "confusión de los modelos."
        )
    st.subheader("Exploración 3D")
    st.pyplot(build_exploration_3d(features, targets), use_container_width=True)
    plt.close("all")
    st.caption(
        "Proyección Tridimensional: Al visualizar tres características "
        "simultáneamente, se aprecia la profundidad de los clústeres. La "
        "separación es más evidente, justificando por qué los modelos logran "
        "precisiones altas al procesar el hiperespacio completo."
    )


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
    if selected_name == "C4.5 (Árbol de Decisión)":
        st.info(
            "Este algoritmo divide el espacio realizando cortes ortogonales "
            "(rectos) buscando la mayor ganancia de información y reducción "
            "de impureza en cada nodo."
        )
    elif selected_name == "K-Nearest Neighbors (KNN)":
        st.info(
            "Al ser un modelo basado en instancias, no crea una abstracción "
            "matemática rígida, sino que traza fronteras adaptativas "
            "dependiendo de la clase mayoritaria de los vecinos más cercanos."
        )
    else:
        st.info(
            "Este clasificador probabilístico traza curvas de decisión "
            "suaves calculando la probabilidad estadística de cada clase, "
            "asumiendo ingenuamente que todas las variables son "
            "independientes."
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
    st.markdown(
        "**Análisis de Aciertos y Fallos:** La diagonal principal (cuadros "
        "oscuros) cuantifica las clasificaciones correctas. Los errores "
        "(fuera de la diagonal) se ubican típicamente en la intersección "
        "predictiva entre Versicolor y Virginica debido a su extrema "
        "similitud biológica en las medidas del sépalo."
    )
    st.subheader("Matriz de Confusión")
    st.pyplot(
        build_confusion_matrix(model, X_test, y_test),
        use_container_width=True,
    )
    plt.close("all")
    if selected_name == "C4.5 (Árbol de Decisión)":
        st.subheader("Importancia de las características")
        st.markdown(
            "**Ganancia de Información:** Este diagrama cuantifica "
            "matemáticamente qué variables aportan más peso a la decisión. "
            "Confirma que el largo y ancho del pétalo dominan la estructura "
            "del árbol, reduciendo la entropía del sistema de manera mucho "
            "más eficiente que los sépalos."
        )
        st.pyplot(
            build_feature_importance(model, X_test, y_test),
            use_container_width=True,
        )
        plt.close("all")
    st.markdown(
        "**Mapeo Geométrico:** Estas áreas de color proyectan el 'territorio' "
        "asignado a cada clase en un plano 2D. Las cruces rojas (**X**) "
        "marcan los datos de prueba clasificados incorrectamente, ilustrando "
        "cómo la superposición espacial engaña a la lógica del algoritmo."
    )
    st.subheader("Frontera de Decisión: Pétalos")
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
    st.subheader("Frontera de Decisión: Sépalos")
    st.pyplot(
        build_decision_boundary(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            feature_pair=("sepal length", "sepal width"),
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
