# ML-Algos
This repository contains implementations of various machine learning algorithms in Python. Each algorithm is implemented from scratch without using high-level libraries like Scikit-Learn, to help understand the underlying mathematics and logic.

# Table of Contents
## **Supervised Algorithms**
1. Linear Regression
2. Logistic Regression
3. Polynomial Regression
4. Lasso Regression
5. Ridge Regression
6. Elastic Net Regression
7. Support Vector Machine


# Supervised Algorithms
## **Linear Regression**
Linear regression is a simple algorithm used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data. 
### Implementation
The `LinearRegression` class is implemented from scratch using NumPy for matrix operations. The implementation includes methods for training the model using gradient descent and making predictions.

## Logistic Regression
Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable. It is a fundamental technique in binary classification tasks.
### Implementation
The `LogisticRegression` class is implemented from scratch using NumPy for matrix operations. The implementation includes methods for training the model using gradient descent, making predictions, and a helper function for the sigmoid function.

## Polynomial Regression
Polynomial regression is a form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modeled as an nth degree polynomial.
### Implementation
The `PolynomialFeatures` class generates polynomial and interaction features. The `PolynomialRegression` class performs polynomial regression using gradient descent.

## Lasso Regression
Lasso regression, or L1 regularization, adds a penalty equal to the absolute value of the magnitude of coefficients to the loss function. It helps in feature selection by shrinking some coefficients to zero.
### Implementation
The LassoRegression class is implemented using gradient descent with L1 regularization.

## Ridge Regression
Ridge Regression, also known as Tikhonov regularization, extends linear regression by introducing a regularization term, which is the L2 penalty. This technique helps in reducing model complexity and multicollinearity, making it more robust for prediction tasks.
### Implementation
1. Gradient Descent Optimization: Utilizes gradient descent to minimize the cost function.
2. L2 Regularization: Includes an L2 penalty term controlled by the alpha parameter.

## Elastic Net Regression
Elastic Net is a regularized regression method that linearly combines the L1 and L2 penalties of the Lasso and Ridge methods. It is particularly useful when there are multiple features that are correlated with each other, which are common in high-dimensional datasets.
### Implementation
The Elastic Net Regression is computed using Gradient Descent using and continuously updating the current weights and bias.

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
