# Diabetes Prediction

## Summary 
We implement a user-friendly web application that uses a questionnaire with easy questions in order to predict whether there is a possibility for a person to develop diabetes. This app aims to inform on time people who are at risk to develop diabetes. 

In no case should replace medical examination.

## Introduction

**Dataset** The data we are using comes from the National Health and Nutrition Examination Survey (NHANES) program at the CDC. NHANES is a research program designed to assess the health and nutritional status of adults and children in the United States. The survey is one of the only to combine both survey questions and physical examinations. It began in the 1960s and since 1999 examines a nationally representative sample of about 5,000 people each year. The NHANES interview includes demographic, socioeconomic, dietary, and health-related questions. The physical exam includes medical, dental, and physiological measurements, as well as several standard laboratory tests. NHANES is used to determine the prevalence of major diseases and risk factors for those diseases. We focus on the Diabetes related data in order to form a prediction model.

Nhanes dataset consists of 31 columns relevant with demographic, socioeconomic, dietary, and health-related informations of 5000 individuals.

The dataset can be found [here](https://data.world/cdc/nhanes)

## Data Preprocessing 
- Handle missing values
- Handle categorical data
- Check correlation between target variable and predictors
- Check correlation between columns
- Select features
- Split train and test data
- Handle imbalanced data


## Model fitting
- Decission Tree (gini and entropy)
- Naive Bayes
- Support Vector Machine
- Ensemble of SVM, Decission tree(with entropy), Naive Bayes
- Multilayer Perceptron
