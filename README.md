# CSE523-Machine-Learning-2022-Data-Dynamos-

Machine Learning Model for Predicting whether ordered item would be fit or not.

This repository contains a machine learning model that uses a dataset of details of females that ordered clothes online  to predict whether the ordered product would be appropriate for them or not. The model performs exploratory data analysis (EDA) on the dataset and applies oversampling techniques to handle class imbalance. The model then uses k-nearest neighbors (KNN) and random forest classifiers (RFC) to make predictions, and grid search cross-validation (GridSearchCV) to optimize the hyperparameters and improve model accuracy.

Dataset

The dataset used in this model contains details of orders of around 1,92,000 women. The dataset is provided in JSON format and includes the following  features:
•	

Exploratory Data Analysis (EDA)

Before building the machine learning model, the dataset is analyzed using EDA techniques to understand the distribution of the features and to identify any outliers or missing values. The EDA also includes visualizations of the dataset using histograms, scatter plots, correlation matrix and box plots.

Oversampling

The dataset contains class imbalance, where some fit types are more prevalent than others. To handle this, oversampling techniques are applied to create synthetic data points for the minority classes, so that the dataset is more balanced and the model can learn to predict the answers effectively.

Machine Learning Model

The machine learning model uses KNN and RFC classifiers to predict body shape rating based on the given input features. KNN is a non-parametric algorithm that uses a distance metric to find the k-nearest neighbors of a data point and assigns the class label based on the majority vote of those neighbors. RFC is an ensemble learning algorithm that uses a collection of decision trees to make predictions.

Hyperparameter Optimization

GridSearchCV is used to optimize the hyperparameters of the KNN and RFC classifiers to improve the accuracy of the model. GridSearchCV is a method that exhaustively searches over a specified parameter space to find the optimal combination of hyperparameters for the given model.

Repository Structure

The repository is structured as follows:

Dataset link/: contains the link to the JSON file containing the dataset
              https://drive.google.com/file/d/11gFGDv03PIOBRrlp-ykVTNZGPcCVEGCp/view?usp=sharing
codes/: contains the python notebooks  used for EDA and model building
Reports/: contain the reports and explanation for the model
Results/: contains the screenshots of the results of the codes
Presentations/: contain the presentations prepared for the same
README.md: this file
Conclusion
This machine learning model provides an accurate prediction of female body shape rating based on a set of input features. By using EDA, oversampling, and hyperparameter optimization techniques, the model can handle class imbalance and achieve high accuracy.

