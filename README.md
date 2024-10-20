# Fraud Detection Project

## Overview

This project aims to analyze and detect fraudulent transactions in e-commerce data. The analysis involves cleaning and preprocessing multiple datasets, performing exploratory data analysis (EDA), and developing machine learning models for fraud detection.

## Key Features

- **Data Preprocessing**: Techniques to clean and prepare data for analysis, ensuring accuracy and reliability.
- **Exploratory Data Analysis (EDA)**: Insights into transaction patterns, fraud indicators, and data distributions.
- **Machine Learning Models**: Implementation of various algorithms to classify transactions as fraudulent or legitimate, including logistic regression, decision trees, and ensemble methods.
- **Evaluation Metrics**: Use of precision, recall, and F1-score to assess model performance and effectiveness in real-world scenarios.
- **Deployment Ready**: Tools and frameworks for deploying models in a production environment to facilitate real-time fraud detection.

## Datasets Used

The analysis utilizes three primary datasets:

1. **Fraud_Data.csv**: Contains e-commerce transactions with features such as:
    - `user_id`: Unique user identifier.
    - `signup_time`: Timestamp of user registration.
    - `purchase_time`: Timestamp of transaction.
    - `purchase_value`: Transaction amount.
    - `device_id`: Identifier for the device used.
    - `class`: Target variable (1 for fraudulent, 0 for non-fraudulent).

2. **IpAddress_to_Country.csv**: Maps IP addresses to countries, including:
    - `lower_bound_ip_address`: Lower bound of IP address range.
    - `upper_bound_ip_address`: Upper bound of IP address range.
    - `country`: Corresponding country.

3. **CreditCard.csv**: Contains bank transaction data featuring:
    - `Time`: Time elapsed since the first transaction.
    - `V1` to `V28`: Anonymized features.
    - `Amount`: Transaction amount.
    - `Class`: Target variable (1 for fraudulent, 0 for non-fraudulent).

## Key Steps Taken

### 1. Data Analysis and Pre-processing

#### Handling Missing Values:
- **Imputation Techniques**: Applied to maintain dataset integrity, filling in gaps based on data distribution and type.
- **Dropping Features**: Features with over 30% missing values were considered for removal, ensuring the dataset remained informative.

#### Data Cleaning:
- **Removing Duplicates**: Identified and eliminated duplicate entries based on key attributes (e.g., transaction identifiers, user IDs) to ensure each transaction was unique.
- **Correcting Data Types**:
    - Identified and converted incorrect data types (e.g., numerical strings to floats, categorical variables to factors).
    - Ensured date and time features were formatted appropriately for analysis.

#### Exploratory Data Analysis (EDA):
- **Univariate Analysis**: Analyzed individual feature distributions.
- **Bivariate Analysis**: Explored relationships between features, particularly concerning the target variable.

#### Geolocation Analysis:
- Merged `Fraud_Data.csv` with `IpAddress_to_Country.csv` to enhance the dataset with geolocation information.

#### Feature Engineering:
- Created new features, including:
    - **Transaction Frequency**: Number of transactions per user.
    - **Transaction Velocity**: Average time between transactions.
    - **Time-Based Features**: Derived features such as `hour_of_day` and `day_of_week`.

#### Normalization and Scaling:
- Applied normalization techniques to ensure equitable contribution of all features during model training.

#### Encoding Categorical Features:
- Transformed categorical features into numerical values to facilitate model training.

Data cleaning is essential in preparing datasets for analysis, particularly in fraud detection, where data quality significantly impacts machine learning performance and insights. This report summarizes the data cleaning process applied to the fraud detection datasets, focusing on duplicate removal and data type correction.

### Data Cleaning Process

#### Removing Duplicates:
- **Identification**: Analyzed datasets to find duplicates based on key attributes.
- **Removal**: Eliminated redundant entries to maintain unique transactions, significantly reducing dataset size for all three datasets.

#### Correcting Data Types:
- **Identification**: Examined features for incorrect data types.
- **Conversion**: Made appropriate conversions, such as:
    - Numerical features converted from string to numeric format.
    - Categorical variables adjusted for better analysis.
    - Date and time features reformatted for time-series analysis.

### Exploratory Data Analysis (EDA) Insights

- **Class Distribution**: Notable imbalance in target variables, with non-fraudulent transactions significantly outnumbering fraudulent ones.
- **Transaction Amount Distribution**: Concentration of lower-value transactions, with a tail toward higher-value purchases in both datasets.
- **Age Distribution**: Younger users are more frequently associated with fraudulent transactions, indicating potential demographic vulnerabilities.
- **Device ID Analysis**: Specific devices linked to multiple fraudulent transactions, highlighting the importance of monitoring device-specific patterns.
- **Geolocation Insights**: Certain countries exhibit higher instances of fraud, informing regional fraud risk assessments.

### Merge Datasets for Geolocation Analysis

- **Loading Datasets**: Loaded cleaned datasets into DataFrames for analysis.
- **IP Address Conversion**: Converted IP addresses to integer format for numerical operations.
- **Merging Datasets**: Performed a left join to enrich the fraud dataset with geographical information.
- **Saving Merged Dataset**: The enriched dataset was saved for future analysis.

### Feature Engineering

- **Load Datasets**: Loaded cleaned fraud and merged datasets.
- **Datetime Conversion**: Converted relevant columns to datetime format.
- **Creating New Features**:
    - `transaction_count`: Number of transactions per user.
    - `transaction_velocity`: Average time between transactions.
    - `hour_of_day`: Extracted from `purchase_time`.
    - `day_of_week`: Captured from `purchase_time`.
- **Saving Updated Dataset**: The updated dataset was saved for subsequent analysis.

### Normalization and Scaling

- **Loading Datasets**: Loaded datasets with engineered features.
- **Feature Selection**: Selected relevant features for normalization and scaling to prepare for model training.

## Conclusion

The data preprocessing steps outlined above ensure that the datasets are of high quality and suitable for further analysis and modeling. Proper data handling, cleaning, and transformation are critical in building effective fraud detection models and deriving actionable insights from the data.

## Installation

To get started with the Fraud Detection Project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/HaYyu-Ra/fraud_detection_project.git
Navigate to the project directory:

bash
Copy code
cd fraud_detection_project
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
To run the project

Contributing
Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For any inquiries or suggestions, feel free to reach out at [hayyu.ragea@gmail.com].
