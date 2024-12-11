# Customer Churn Prediction

## Overview

Customer churn prediction is a crucial challenge for the banking sector as retaining high-value customers directly impacts profitability and operational efficiency. This project develops a robust machine learning-based solution to predict customer churn using a dataset of 10,000 bank customers. The solution incorporates data preprocessing techniques, including feature encoding, normalization, and class balancing using SMOTE, along with exploratory data analysis (EDA) for understanding customer behavior. Four machine learning models—Decision Tree, Random Forest, Gradient Boosting, and a Stacking Ensemble—were employed. The stacking ensemble combines the predictions of Decision Tree, Random Forest, and Gradient Boosting models as base learners, using Logistic Regression as the meta-classifier. This approach achieved the highest accuracy of 90% and an AUC-ROC score of 0.95. The project highlights the importance of ensemble methods and advanced preprocessing techniques in improving prediction accuracy and demonstrates deployment readiness through integration with Google Cloud Platform for real-time predictions.

## Objective

The objective of this project is to develop a scalable and accurate machine learning model to predict customer churn in the banking sector. The focus is on balancing interpretability, performance, and deployability to empower banks with actionable insights for customer retention.

## Key Features

- **Data Preprocessing**: Techniques like outlier removal, feature scaling, one-hot encoding, and SMOTE for class imbalance correction.
- **Exploratory Data Analysis (EDA)**: Insights into churn behavior, including correlations with demographic and transactional features.
- **Machine Learning Models**: Implementation of Decision Tree, Random Forest, Gradient Boosting, and a Stacking Ensemble.
- **Stacking Ensemble**: Combines the strengths of base learners (Decision Tree, Random Forest, Gradient Boosting) with a Logistic Regression meta-classifier for superior predictive performance.
- **Deployment**: Integration with Google Cloud Platform (GCP) for real-time predictions.

## Dataset

- **Source**: Kaggle
- **Description**: A dataset containing 10,000 records with demographic, behavioral, and transactional attributes, labeled to indicate churn.
- **Features**:
  - Demographic: `Geography`, `Gender`, `Age`
  - Behavioral: `IsActiveMember`, `NumOfProducts`
  - Transactional: `CreditScore`, `Balance`, `EstimatedSalary`
- **Target Variable**: `Exited` (1 = Churned, 0 = Stayed)

## Preprocessing Steps

1. **Outlier Detection and Removal**: Using IQR for features like `Balance` and `Age`.
2. **Feature Engineering**: One-hot encoding for categorical variables and normalization for numerical features.
3. **Class Balancing**: SMOTE to generate synthetic samples for the minority class.
4. **Data Splitting**: Stratified train-test split (80%-20%).

## Models and Results

### Decision Tree
- **Accuracy**: 73%
- **Hyperparameters**: `max_depth=5`, `criterion='gini'`

### Random Forest
- **Accuracy**: 82%
- **Hyperparameters**: `n_estimators=10`, `max_depth=5`

### Gradient Boosting
- **Accuracy**: 73%
- **Hyperparameters**: `n_estimators=100`, `learning_rate=0.01`

### Stacking Ensemble
- **Components**:
  - **Base Learners**: Decision Tree, Random Forest, Gradient Boosting
  - **Meta-Classifier**: Logistic Regression with L2 regularization
- **Performance**:
  - **Accuracy**: 90%
  - **AUC-ROC**: 0.95

## Performance Comparison

| Model                | Accuracy | AUC-ROC |
|----------------------|----------|---------|
| Decision Tree        | 73%      | 0.73    |
| Random Forest        | 82%      | 0.82    |
| Gradient Boosting    | 73%      | 0.73    |
| Stacking Ensemble    | 90%      | 0.95    |

## Repository Structure

- `churn.csv`: Original dataset.
- `preprocessed_with_smote.csv`: Balanced dataset after preprocessing.
- `preprocess.ipynb`: Notebook for data preprocessing and EDA.
- `decision_tree_implementation.ipynb`: Decision Tree model implementation.
- `GBoost.ipynb`: Gradient Boosting model implementation.
- `stacking.ipynb`: Stacking Ensemble model implementation.
- `README.md`: Project documentation.

## How to Run

 Clone the repository:
   ```bash
   git clone <repository_url>
   cd Customer_Churn_Prediction
  ```
## Insights

### Key Drivers of Churn
- **Inactivity**: Customers with `IsActiveMember = 0` have a significantly higher likelihood of churning.
- **Regional Differences**: Higher churn rates were observed in Germany compared to other regions.
- **Customer Behavior**: Customers with low account balances or shorter tenures are more prone to churn.

### Model Advantages
- **Ensemble Methods**: Techniques like stacking provide better performance and robustness by combining multiple models.
- **Advanced Preprocessing**: Methods such as SMOTE and feature scaling enhance the accuracy and interpretability of the models.

## Contact

For any questions or collaboration, feel free to reach out:

- **Abhijay Rane**: ar2536@cornell.edu  
- **Shubham Gandhi**: smg384@cornell.edu
