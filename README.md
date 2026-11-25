Machine Learning Assignment 2

Dataset
Adult Census Income Dataset

Objective
Predict whether an individual's annual income exceeds $50,000.

Dataset Statistics
Records: 32,561
Features: 14
Target Variable: income

Preprocessing
Missing value handling
Duplicate removal
One-Hot Encoding
Standard Scaling
Stratified Train-Test Split (80:20)

Models Used
Logistic Regression
Decision Tree
KNN
Random Forest
Naive Bayes

Results
Model	            Accuracy  AUC	Precision Recall  F1	MCC
Logistic Regression	0.8502	0.9004	0.7344	0.5925	0.6558	0.5668
Decision Tree	    0.8099	0.7407	0.6052	0.6071	0.6062	0.4809
KNN	                0.8274	0.8510	0.6618	0.5804	0.6184	0.5094
Random Forest   	0.8494	0.8942	0.7204	0.6129	0.6623	0.5693
Naive Bayes	0.5244	0.7218	0.3305	0.9496	0.4904	0.3142

Best Model
Random Forest

Conclusion
Random Forest achieved the highest F1 Score and MCC while maintaining strong Accuracy and AUC values. Therefore, it is selected as the best model for this classification problem.