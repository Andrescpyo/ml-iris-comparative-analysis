<div align="center">

# 📊 Análisis Comparativo de Clasificadores ML - Dataset Iris

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

</div>

## 📝 Descripción del Proyecto

Este proyecto de ingeniería, desarrollado para la **Universidad Distrital Francisco José de Caldas**, tiene como objetivo entrenar, evaluar y comparar empíricamente tres algoritmos fundamentales de clasificación:

- **C4.5**, implementado mediante un árbol de decisión basado en entropía.
- **K-Nearest Neighbors (KNN)**.
- **Naive Bayes**.

El análisis utiliza el clásico dataset Iris para estudiar el comportamiento de los modelos a partir de sus métricas de precisión, tiempos computacionales, fronteras de decisión e importancia de características.

## ✨ Características Principales

- 🖥️ Interfaz web interactiva desarrollada con Streamlit.
- 📈 Comparación de rendimiento entre tiempos de entrenamiento, tiempos de predicción y precisión.
- 🗺️ Trazado dinámico de fronteras de decisión para las características de pétalos y sépalos.
- 🔥 Visualización de matrices de confusión y errores de clasificación.
- 📊 Gráficos exploratorios 2D y 3D del dataset Iris.
- 🌸 Simulador de inferencia en vivo para clasificar nuevas flores mediante sus cuatro características biológicas.
- 🎛️ Ajuste interactivo de hiperparámetros para KNN y el árbol de decisión.
- 🧮 Análisis de importancia de características mediante reducción de entropía y permutación.

## 🛠️ Tecnologías Utilizadas

- 🐍 **Python**
- 🤖 **Scikit-Learn**
- 🐼 **Pandas**
- 📉 **Matplotlib**
- 🎨 **Seaborn**
- 🚀 **Streamlit**
- 🧪 **ucimlrepo**, para obtener el dataset Iris desde el repositorio UCI

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd ml-iris-comparative-analysis
```

### 2. Crear el entorno virtual

```bash
python -m venv venv
```

### 3. Activar el entorno virtual

En Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

En macOS o Linux:

```bash
source venv/bin/activate
```

### 4. Instalar las dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá en el navegador mediante la URL local mostrada por Streamlit.

## 🗂️ Estructura del Proyecto

```text
ml-iris-comparative-analysis/
├── app.py
│   └── Interfaz Streamlit, navegación y visualizaciones interactivas
├── requirements.txt
│   └── Dependencias del proyecto
├── .gitignore
│   └── Archivos temporales, entornos virtuales y datos locales
├── src/
│   ├── data_loader.py
│   │   └── Carga y partición reproducible del dataset Iris
│   ├── classifier_factory.py
│   │   └── Creación parametrizable de los clasificadores
│   ├── evaluator.py
│   │   └── Evaluación de precisión y tiempos de ejecución
│   └── main.py
│       └── Ejecución del análisis comparativo por consola
└── notebooks/
    └── visualizacion.ipynb
        └── Exploración gráfica reproducible del dataset
```

## 👤 Autor

**Andrés Cerdas Padilla**  
Universidad Distrital Francisco José de Caldas
