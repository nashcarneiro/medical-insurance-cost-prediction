Medical Insurance Cost Prediction

# Overview
This project analyzes a medical insurance dataset to explore the factors that influence insurance charges and develops multiple regression models to predict medical costs.

The project covers the complete workflow from data cleaning and exploratory data analysis to model development, refinement, evaluation, and visual comparison. Multiple regression approaches are compared using the R² score, and the best-performing model is further evaluated by comparing its predicted charges against actual values from the test dataset.

# Project Objectives

The main objectives of this project are to:

* Clean and preprocess a medical insurance dataset.
* Handle missing values using appropriate imputation methods.
* Explore relationships between patient attributes and medical insurance charges.
* Identify which features have the strongest correlation with insurance costs.
* Develop and evaluate multiple regression models.
* Compare model performance using the R² score.
* Visualize actual versus predicted charges for the best-performing model.

# Dataset

The dataset contains information about individuals and their corresponding medical insurance charges.

The features include:

* age — Age of the individual.
* gender — Gender of the individual.
* bmi — Body mass index.
* no_of_children — Number of children or dependents.
* smoker — Smoking status.
* region — Geographic region.
* charges — Medical insurance charges.

The original CSV file does not contain column headers, so descriptive column names are manually assigned after loading the dataset.

# Technologies and Libraries

The project was developed using Python and the following libraries:

* Pandas — Data loading, cleaning, and manipulation.
* NumPy — Numerical operations and missing-value handling.
* Matplotlib — Data visualization.
* Seaborn — Statistical visualization.
* Scikit-learn — Data splitting, preprocessing, regression modeling, regularization, pipelines, prediction, and model evaluation.

# Project Workflow

# 1. Data Loading and Preparation

The dataset is loaded using Pandas without predefined column headers. Descriptive column names are then assigned to improve readability and simplify further analysis.

Missing values represented by `?` are replaced with `NaN` to allow proper detection and handling.

# 2. Handling Missing Values

Missing values are handled using different strategies depending on the feature:

* Missing `age` values are replaced with the mean age.
* Missing `smoker` values are replaced with the most frequently occurring smoking status.

Insurance charges are rounded to two decimal places for consistency.

# Exploratory Data Analysis

# BMI vs Medical Charges

A regression plot is used to explore the relationship between BMI and medical insurance charges.

This visualization helps identify whether higher BMI values are associated with changes in medical costs and reveals the overall pattern and spread of the data.

# Smoking Status vs Medical Charges

A box plot compares the distribution of medical charges between smoking categories.

The analysis shows that smoking status has a strong relationship with insurance charges and is the most highly correlated feature among the selected predictors.

# Feature Correlation with Charges

The correlation between selected insurance features and medical charges is calculated and visualized using a horizontal bar chart.

The analyzed features are:

* Age
* BMI
* Number of children
* Smoking status
* Region

This visualization makes it easier to compare the strength and direction of each feature's relationship with medical charges.

# Model Development

Five regression approaches are developed and evaluated in this project.

# Simple Linear Regression

A Linear Regression model is developed using only smoking status as the predictor because it demonstrates the strongest correlation with medical insurance charges among the selected features.

# Multiple Linear Regression

A second Linear Regression model uses multiple attributes:

* Age
* BMI
* Number of children
* Smoking status
* Region

The model is trained using 80% of the dataset and evaluated on the remaining 20% test set.

# Polynomial Regression Pipeline

A Scikit-learn pipeline is created with three stages:

1. Feature standardization using StandardScaler.
2. Polynomial feature generation using PolynomialFeatures.
3. Prediction using LinearRegression.

This approach allows the model to capture nonlinear relationships and interactions between insurance features.

# Ridge Regression

A Ridge Regression model is trained using a regularization parameter of alpha=0.1.

Ridge regularization helps control model complexity by penalizing large model coefficients.

# Polynomial Ridge Regression

The original features are transformed into second-degree polynomial features before fitting a Ridge Regression model.

This approach combines nonlinear feature interactions with regularization and is evaluated using unseen test data.

# Model Performance Comparison

The following models are compared using their R² scores:

* Simple Linear Regression
* Multiple Linear Regression
* Polynomial Regression Pipeline
* Ridge Regression
* Polynomial Ridge Regression

A horizontal bar chart is used to visually compare model performance and identify which regression approaches explain the greatest proportion of variation in medical insurance charges.

# Actual vs Predicted Charges

The predictions from the best-performing model are compared against the actual insurance charges in the test dataset using a scatter plot.

Each point represents one observation:

* The x-axis represents the actual medical charge.
* The y-axis represents the model's predicted medical charge.
* The diagonal dashed line represents perfect predictions, where the predicted value exactly matches the actual value.

Points closer to the diagonal line represent more accurate predictions, while points farther from the line indicate larger prediction errors.

# Key Insights

* Smoking status shows the strongest correlation with medical insurance charges among the selected predictors.
* Using multiple features provides more information for predicting insurance costs than relying on a single predictor.
* Polynomial transformations allow regression models to capture nonlinear relationships and interactions between features.
* Polynomial-based regression approaches achieved the strongest model performance in this analysis.
* The actual-versus-predicted visualization provides a direct view of how closely the selected model's predictions match real insurance charges.
