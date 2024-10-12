<img src="https://raw.githubusercontent.com/ChanLumerico/luma/main/img/title/dark.png" alt="logo" height="50%" width="50%">

A Comprehensive Python Module for Machine Learning and Data Science

<img alt="pypi-version" src="https://img.shields.io/pypi/v/luma-ml?logo=python&logoColor=white&color=blue">
<img alt="pypi-downloads" src="https://img.shields.io/pypi/dm/luma-ml">
<img src="https://img.shields.io/badge/total downloads-12.8k-red">
<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/ChanLumerico/luma?color=yellow">
<img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-000000.svg">

## About

Luma is a comprehensive, user-friendly Python library designed for both beginners
and advanced users in the field of machine learning and data science. It provides
a wide range of tools and functionalities to streamline the process of data analysis,
model building, evaluation, and deployment.

### Purpose

Luma is built for an educational purpose, focused on implementing various machine learning algorithms and models **from scratch** solely depending on low-level libraries such as `NumPy`.

### Key Features

- **Easy Data Handling**: Simplify data preprocessing, transformation, and visualization.
- **Model Building**: Access a variety of machine learning algorithms and models.
- **Model Evaluation**: Utilize robust tools for model validation and tuning.

## Packages

| Name | Description |
| --- | --- |
| `luma.classifier` | Toolkit for classification models including various algorithms. |
| `luma.clustering` | Focuses on unsupervised learning and clustering algorithms. |
| `luma.core` | Foundational backbone providing essential data structures and utilities. |
| `luma.ensemble` | Ensemble learning methods for improved model performance. |
| `luma.extension` | Various extensions for Luma development. Not for end-users. |
| `luma.interface` | Protocols and custom data types for internal use within Luma. |
| `luma.metric` | Performance metrics for evaluating machine learning models. |
| `luma.migrate` | Import and export of machine learning models within Luma. |
| `luma.model_selection` | Tools for model selection and hyperparameter tuning. |
| `luma.neural` [🔗](https://github.com/ChanLumerico/luma-neural) | Deep learning models and neural network utilities. A dedicated DL package for Luma. |
| `luma.pipe` | Creating and managing machine learning pipelines. |
| `luma.preprocessing` | Data preprocessing functions for machine learning tasks. |
| `luma.reduction` | Dimensionality reduction techniques for high-dimensional datasets. |
| `luma.regressor` | Comprehensive range of regression algorithms. |
| `luma.visual` | Tools for model visualization and data plotting. |

---

## Getting Started

### Installation

To get started with Luma, install the package using `pip`:

```bash
pip install luma-ml
```

Or for a specific version,

```bash
pip install luma-ml==[any_version]
```

### Import

After installation, import Luma in your Python script to access its features:

```python
import luma
```

### Acceleration

Luma supports `MLX` based `NumPy` acceleration on **Apple Silicon**. By importing Luma’s neural package, it will automatically detect Apple’s Metal Performance Shader(MPS) availability and directly apply MLX acceleration for all execution flows and operations using `luma.neural`.

```python
import luma.neural
```

Otherwise, the default CPU based operation is applied.

For more details, please refer to the link 🔗 shown at Luma’s neural package description.

---

## Others

### Contribution

Luma is an open-source project, and we welcome contributions from the community. 😃

Whether you're interested in fixing bugs, adding new features, or improving documentation, your help is appreciated.

### License

Luma is released under the GPL-3.0 License. See `LICENSE` file for more details.

### Inspired By

Luma is inspired by these libraries:

<img src="https://skillicons.dev/icons?i=sklearn,pytorch,tensorflow">

### Specifications

| | Description |
| --- | --- |
| Latest Version | 1.2.2 |
| Lines of Code | ~40.4K |
| Dependencies | NumPy, SciPy, Pandas, Matplotlib, Seaborn, MLX(Optional) |
