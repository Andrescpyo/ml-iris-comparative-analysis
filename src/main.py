"""Run the comparative classification analysis on the Iris dataset."""

from data_loader import get_iris_data
from classifier_factory import create_classifiers
from evaluator import evaluate_model


def main() -> None:
    """Load data, evaluate every classifier, and print a results table."""
    X_train, X_test, y_train, y_test = get_iris_data()
    classifiers = create_classifiers()
    results = [
        evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        for name, model in classifiers.items()
    ]

    print(
        f"{'Nombre':<30} {'Precisión':>10} "
        f"{'Tiempo de Entrenamiento (s)':>30} "
        f"{'Tiempo de Predicción (s)':>27}"
    )
    print("-" * 100)
    for result in results:
        print(
            f"{result['name']:<30} "
            f"{result['accuracy']:>10.4f} "
            f"{result['training_time']:>30.6f} "
            f"{result['prediction_time']:>27.6f}"
        )


if __name__ == "__main__":
    main()
