# Titanic Survival Prediction — End-to-End Machine Learning

## Project Overview

This project develops and evaluates machine learning models to predict passenger survival aboard the RMS Titanic using the [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) dataset.

The project follows an end-to-end machine learning workflow, including exploratory data analysis (EDA), data preprocessing, missing-value imputation, feature engineering, model development, hyperparameter tuning, cross-validation, feature importance analysis, and Kaggle submission.

**Final selected model:** XGBoost Classifier

**Kaggle Public Leaderboard Accuracy: 76.315%**

## Dataset

The dataset contains passenger demographic, socioeconomic, and travel-related information.

- Training dataset: 891 passengers, 12 columns
- Test dataset: 418 passengers, 11 columns
- Target variable: `Survived` (0 = Did Not Survive, 1 = Survived)

Key features include passenger class, sex, age, fare, family relationships, embarkation port, passenger names, tickets, and cabin information.

## Methodology

### 1. Exploratory Data Analysis

Explored survival patterns across passenger sex, class, age, family size, fare, embarkation port, and cabin availability.

Key observations included substantially higher survival rates among women and first-class passengers, along with differences associated with family size and passenger age.

### 2. Data Preprocessing

- Analyzed missing values in `Age`, `Cabin`, `Embarked`, and `Fare`.
- Developed grouped-median age imputation using sex, passenger class, parental relationships, and extracted passenger titles.
- Applied fare imputation using passenger-class medians.
- Encoded categorical variables and standardized numerical features.
- Implemented custom scikit-learn transformers to fit preprocessing operations within cross-validation folds and reduce data leakage.

### 3. Feature Engineering

Developed features including:

- `Title` — extracted from passenger names
- `FamilySize` — number of family members traveling together
- `IsAlone` — indicator for passengers traveling alone
- `AgeMissing` — missing-age indicator
- `HasParch` — indicator for parent/child relationships
- `CabinKnown` — indicator for cabin information availability
- `LogFare` — logarithmic transformation of passenger fare

Additional experiments investigated surname and ticket group sizes, fare per passenger, and family-related characteristics.

### 4. Model Development

Compared several classification algorithms:

| Model | 5-Fold CV Accuracy |
|---|---:|
| Logistic Regression | 83.05% |
| Random Forest | 82.71% |
| XGBoost | **84.62%** |
| CatBoost | 84.17% |

Evaluation metrics included accuracy, precision, recall, F1-score, and ROC-AUC.

### 5. Hyperparameter Tuning and Validation

Performed randomized hyperparameter search for XGBoost and evaluated model generalization using:

- Stratified K-Fold Cross-Validation
- Repeated Stratified Cross-Validation
- Ticket-grouped cross-validation
- Nested cross-validation for tuning evaluation
- Model complexity and feature ablation experiments

The original XGBoost configuration was retained after the tuned model did not demonstrate a reliable improvement under nested validation.

### 6. Model Interpretation and Error Analysis

Investigated model behavior using permutation feature importance, confusion matrices, out-of-fold predictions, probability calibration, and prediction confidence.

The most influential raw inputs included passenger name (through title extraction), sex, passenger class, age, and family relationships.

Classification errors were particularly concentrated among third-class female passengers and first-class male passengers.

### 7. Additional Feature Experiments

Tested ticket-sharing and family-based features, including `FarePerPerson`.

Although `FarePerPerson` improved repeated cross-validation accuracy from 83.52% to 84.38%, the Kaggle public leaderboard score declined from 76.315% to 75.119%.

This demonstrated the importance of evaluating generalization rather than selecting models based solely on improvements in local cross-validation metrics.

## Final Results

| Metric | Original XGBoost |
|---|---:|
| 5-Fold CV Accuracy | 84.62% |
| Repeated CV Accuracy | 83.52% |
| Ticket-Grouped CV Accuracy | 82.61% |
| 5-Fold CV Precision | 81.78% |
| 5-Fold CV Recall | 77.20% |
| 5-Fold CV F1-Score | 79.40% |
| 5-Fold CV ROC-AUC | 88.38% |
| Kaggle Public Leaderboard Accuracy | **76.315%** |

The difference between local validation and Kaggle performance highlights the importance of robust validation, generalization analysis, and cautious feature selection.

## Technologies Used

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, CatBoost, and Jupyter Notebook.

## Key Takeaways

- Feature engineering can improve model performance, but gains must be validated on unseen data.
- Cross-validation design matters, particularly when observations share family or ticket relationships.
- More complex models and additional features do not automatically improve generalization.
- Model interpretation and error analysis provide valuable insights beyond classification accuracy.
- A reproducible machine learning workflow is as important as the final leaderboard score.

## Data Source

[Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

## Author

Hussein Alim

Machine Learning | Data Science | Predictive Analytics
