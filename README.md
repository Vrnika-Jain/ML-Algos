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
7. Support Vector Machine (SVM)
8. Naive Bayes Classification(s)
9. K Nearest Neighbors (KNN)
10. Artificial Neural Networks (ANN)

## **Unsupervised Algorithms**
1. K Means Clustering
2. Hierarichal Clustering
3. Apriori Algorithm
4. DBSCAN Clustering
5. PAM Clustering


# Supervised Algorithms
## **Linear Regression**
Linear regression is a simple algorithm used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data. 
### Implementation
The `LinearRegression` class is implemented from scratch using NumPy for matrix operations. The implementation includes methods for training the model using gradient descent and making predictions.

## **Logistic Regression**
Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable. It is a fundamental technique in binary classification tasks.
### Implementation
The `LogisticRegression` class is implemented from scratch using NumPy for matrix operations. The implementation includes methods for training the model using gradient descent, making predictions, and a helper function for the sigmoid function.

## **Polynomial Regression**
Polynomial regression is a form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modeled as an nth degree polynomial.
### Implementation
The `PolynomialFeatures` class generates polynomial and interaction features. The `PolynomialRegression` class performs polynomial regression using gradient descent.

## **Lasso Regression**
Lasso regression, or L1 regularization, adds a penalty equal to the absolute value of the magnitude of coefficients to the loss function. It helps in feature selection by shrinking some coefficients to zero.
### Implementation
The `LassoRegression` class is implemented using gradient descent with L1 regularization.

## **Ridge Regression**
Ridge Regression, also known as Tikhonov regularization, extends linear regression by introducing a regularization term, which is the L2 penalty. This technique helps in reducing model complexity and multicollinearity, making it more robust for prediction tasks.
### Implementation
1. Gradient Descent Optimization: `Ridge Regression` class Utilizes gradient descent to minimize the cost function.
2. L2 Regularization: Includes an L2 penalty term controlled by the alpha parameter.

## **Elastic Net Regression**
Elastic Net is a regularized regression method that linearly combines the L1 and L2 penalties of the Lasso and Ridge methods. It is particularly useful when there are multiple features that are correlated with each other, which are common in high-dimensional datasets.
### Implementation
The `Elastic Net Regression` is computed using Gradient Descent using and continuously updating the current weights and bias.

## **Support Vector Machine (SVM)**
SVM is a powerful supervised machine learning algorithm that can be used for both classification and regression tasks. It aims to find the optimal hyperplane that best separates the classes in the feature space.
### Implementation
The `Support Vector Machine` (SVM) algorithm works by finding the hyperplane that best separates the data into different classes. The hyperplane is chosen to maximize the margin, which is the distance between the hyperplane and the nearest data points from each class, known as support vectors.

## **Naive Bayes Classification**
Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features. They are particularly popular for text classification and spam filtering.
### Gaussian Naive Bayes
Gaussian Naive Bayes is used for classification with continuous data that follows a normal (Gaussian) distribution.
### Multinomial Naive Bayes
Multinomial Naive Bayes is used for classification with discrete data, particularly for text classification.
### Bernoulli Naive Bayes
Bernoulli Naive Bayes is used for binary/boolean features.

## **K Nearest Neighbors**
KNN is a non-parametric, instance-based learning algorithm that classifies new data points based on the majority label of their K-nearest neighbors in the training data. It can also be used for regression by averaging the target values of the K-nearest neighbors.
### Implementation
The `K Nearest Neighbors` depends on distance between the test point and all training points, and common class label or average target value for classification and regression respectively.

## **Artificial Neural Networks (ANN)**
ANNs are inspired by the structure and function of the human brain and are used to solve complex problems in areas such as classification, regression, and pattern recognition.
### Implementation
For implementing `Artificial Neural Networks`, a dataset with features and target labels, network architecture, learning rate, and number of iterations. Then the output of the network is computed by passing inputs through the layers and Updating weights and biases which were primarily initialized as Zeros.


# Unsupervised Algorithms
## **K Means Clustering**
K Means Clustering technique used for partitioning a dataset into distinct groups (clusters).
### Implemnentation
`K-Means Clustering` is a centroid-based algorithm where the number of clusters (K) is predefined. The algorithm aims to minimize the variance within each cluster, effectively grouping similar data points together.

## **Hierarichal Clustering**
Hierarchical Clustering seeks to build a hierarchy of clusters. It can be either Agglomerative (bottom-up approach) or Divisive (top-down approach). This implementation focuses on Agglomerative Hierarchical Clustering.
### Implementation
The `Hierarichal Clustering` works by calculating the distance between each pair of observations using a distance metric such as Euclidean distance and using linkage matrix to construct a dendrogram, which visually represents the hierarchy of clusters.

## **Apriori Algorithm**
Apriori algorithm, is a classic algorithm used in data mining for learning association rules. Apriori is used to identify frequent itemsets in a dataset and derive the association rules from them.
### Implementation
The `Apriori Algorithm` uses a "bottom-up" approach, where frequent subsets are extended one item at a time (a step known as candidate generation), and groups of candidates are tested against the data. The algorithm terminates when no further successful extensions are found.

## **DBSCAN Clustering**
DBSCAN Algorithm (Density-based spatial clustering) groups together points that are close to each other based on a distance measurement and a minimum number of points. It is particularly effective for finding clusters of arbitrary shape and for identifying outliers (noise).
### Implementation
The `DBSCAN Clustering` depends on Density Reachability and Density Connectivity.


# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
