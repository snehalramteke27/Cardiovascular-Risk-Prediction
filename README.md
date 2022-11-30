# Cardiovascular-Risk-Prediction 

# Introduction:

* Cardiovascular diseases (CVDs) are the major cause of mortality worldwide. According to WHO, 17.9 million people died from CVDs in 2019, accounting for 32% of all global fatalities.
* Though CVDs cannot be treated, predicting the risk of the disease and taking the necessary precautions and medications can help to avoid severe symptoms and, in some cases, even death.
* As a result, it is critical that we accurately predict the risk of heart disease in order to avert as many fatalities as possible.
* The goal of this project is to develop a classification model that can predict if a patient is at risk of coronary heart disease (CHD) over the period of 10 years, based on demographic, lifestyle, and medical history.

# EDA Summary:

* The dependent variable is unbalanced, with only ~15% of the patients who tested positive for CHD.
* All the continuous variables are positively skewed except age (which is almost normally distributed)
* Majority of the patients belong to the education level 1, followed by 2, 3, and 4 respectively.
* There are more female patients compared to male patients.
* Almost half the patients are smokers.
* 100 patients under the study are undertaking blood pressure medication.
* 22 patients under the study have experienced a stroke.
* 1069 patients have hypertension.
* 87 patients have diabetes.
* The risk of CHD is higher for older patients than younger patients.
* 18%, 11%, 12%, 14% of the patients belonging to the education level 1, 2, 3, 4 respectively were eventually diagnosed with CHD.
* Male patients have significantly higher risk of CHD (18%) than female patients (12%)
* Patients who smoke have significantly higher risk of CHD (16%) than patients who don't smoke (13%)
* Patients who take BP medicines have significantly higher risk of CHD (33%) than other patients (14%)
* Patients who had experienced a stroke in their life have significantly higher risk of CHD (45%) than other patients (14%)
* Hypertensive patients have significantly higher risk of CHD (23%) than other patients (11%)
* Diabetic patients have significantly higher risk of CHD (37%) than other patients (14%)
* Above is the correlation heatmap for all the continuous variables in the dataset.
* The variables ‘systolic BP’ and ‘diastolic BP’ are highly correlated.
* To handle high correlation between two independent variables, we can introduce a new variable ‘pulse_pressure’

# Modelling Summary:

1.Logistic Regression:

* In statistics, the (binary) logistic model is a statistical model that models the probability of one event (out of two alternatives) taking place by having the log-odds (the logarithm of the odds) for the event be a linear combination of one or more independent variables.
* This can be considered as the baseline model to obtain predictions since it is easy to explain.
* Logistic Regression train recall: 0.69
* Logistic Regression test recall: 0.66

 2. K-nearest Neighbors:

* The k-nearest neighbors algorithm, also known as KNN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.
* Best hyperparameters: K = 55
* K nearest neighbors train recall: 0.83
* K nearest neighbors test recall: 0.69

3. Naïve Bayes:

* Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e., every pair of features being classified is independent of each other.
* Best hyperparameters: var_smoothing= 0.657933224657568
* Naïve Bayes train recall: 0.53
* Naïve Bayes test recall: 0.50

 4. Decision Tree:

* A Decision tree is a flowchart-like tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label.
* Best hyperparameters: max_depth: 1, min_samples_leaf: 0.1, min_samples_split: 0.1
* Decision tree train recall: 0.86
* Decision tree test recall: 0.77

 5. Support Vector Machine:

* Support Vector Machine (SVM) is a supervised machine learning algorithm used for both classification and regression. The objective of SVM algorithm is to find a hyperplane in an N-dimensional space that distinctly classifies the data points.
* Best hyperparameters: C: 1, gamma: 0.01, kernel: rbf
* SVM train recall: 0.74
* SVM test recall: 0.69

6. XG Boost:

* In this algorithm, decision trees are created in sequential form. Weights play an important role in XGBoost. Weights are assigned to all the independent variables which are then fed into the decision tree which predicts results. The weight of variables predicted wrong by the tree is increased and these variables are then fed to the second decision tree. These individual classifiers/predictors then ensemble to give a strong and more precise model. It can work on regression, classification, ranking, and user-defined prediction problems.
* Best hyperparameters: max_depth: 1, min_samples_leaf: 0.1, min_samples_split: 0.1, n_estimators: 500
* XG boost train recall: 0.78
* XG boost test recall: 0.60

# Conclusion:

* We trained 6 Machine Learning models using the training dataset, and hyperparameter tuning was used in some models to improve the model performance.
* To build the models, missing values were handled, feature engineering and feature selection was performed, and the training dataset was oversampled using SMOTE to reduce bias on one outcome.
* Recall was chosen as the model evaluation metric because it was very important that we reduce the false negatives.
* Initial set of predictions were obtained using the baseline model, ie, logistic regression model, and other commonly used classification models were also build in search of better predictions.
* Predicting the risk of coronary heart disease is critical for reducing fatalities caused by this illness. We can avert deaths by taking the required medications and precautions if we can foresee the danger of this sickness ahead of time.
* It is critical that the model we develop has a high recall score. It is OK if the model incorrectly identifies a healthy patient as a high risk patient because it will not result in death, but if a high risk patient is incorrectly labelled as healthy, it may result in fatality.
* We were able to create a model with a recall of just 0.77 because of limitated data available and limited computational power availabe.
* A recall score of 0.77 indicates that out of 100 individuals with the illness, our model will be able to classify only 77 as high risk patients, while the remaining 33 will be misclassified.
* Future developments must include a strategy to improve the model recall score, enabling us to save even more lives from this disease. This includes involving more people in the study, and include people with different medical history, etc build an application with better recall score.
* From our analysis, it is also found that the age of a person was the most important feature in determining the risk of a patient getting infected with CHD, followed by pulse pressure, prevalent hypertension and total cholesterol.
* Diabetes, prevalent stroke and BP medication were the least important features in determining the risk of CHD.

