# Fraud Detection Project
# Overview
This project focuses on analyzing and detecting fraudulent transactions in e-commerce and credit card data. The key steps involve data preprocessing, exploratory data analysis (EDA), and developing machine learning models to identify fraud. The objective is to improve transaction security by reducing fraudulent activity through accurate detection models.

# Key Features
Data Preprocessing: Techniques to clean and prepare data, ensuring reliability and accuracy.
Exploratory Data Analysis (EDA): Insights into patterns, fraud indicators, and data distributions.
Machine Learning Models: Use of algorithms like logistic regression, decision trees, and ensemble methods to classify transactions as fraudulent or legitimate.
Evaluation Metrics: Performance is evaluated using precision, recall, and F1-score, critical for real-world fraud detection.
Deployment Ready: Integration of tools for real-time fraud detection in production environments.
# Datasets Used
Fraud_Data.csv: E-commerce transactions with features such as user_id, signup_time, purchase_time, purchase_value, device_id, and the target variable class (1 for fraudulent, 0 for non-fraudulent).
IpAddress_to_Country.csv: Maps IP address ranges to countries, adding geographical information to transaction data.
CreditCard.csv: Credit card transaction data with anonymized features (V1 to V28) and the target variable Class (1 for fraudulent, 0 for non-fraudulent).
# Key Steps Taken
1. Data Analysis and Preprocessing
Handling Missing Values: Imputation and dropping features with over 30% missing values.
Data Cleaning: Removal of duplicates, correction of data types, and ensuring the integrity of date/time fields.
Exploratory Data Analysis (EDA): Univariate and bivariate analysis, geolocation insights, and visualization of transaction patterns.
2. Data Cleaning Process
Removing Duplicates: Identified redundant entries and removed duplicates to ensure unique transactions.
Correcting Data Types: Converted numerical strings to appropriate formats and adjusted categorical variables.
3. Exploratory Data Analysis (EDA) Insights
Class Imbalance: Non-fraudulent transactions outnumber fraudulent ones.
Transaction Amounts: Most fraudulent activities involve lower transaction amounts.
Device and Geolocation Analysis: Specific devices and countries show higher instances of fraud, informing regional risk assessments.
4. Merging Datasets for Geolocation Analysis
IP Conversion: Converted IP addresses to integer format and merged datasets for enhanced analysis.
Geographical Insights: Fraud trends analyzed by country, enhancing detection models.
5. Feature Engineering
New Features: Created transaction frequency, velocity, and time-based features like hour_of_day and day_of_week.
Normalization and Scaling: Applied to features for equitable model training.
6. Model Building and Training
Train-Test Split: Utilized train_test_split to create training and test datasets for both fraud and credit card data.
Models Tested: Algorithms such as logistic regression, decision trees, random forest, gradient boosting, multi-layer perceptron (MLP), and recurrent neural networks (RNN) were used.
# Interim 2 Submission: Model Building and Training
# Overview
This report documents the process of building and training machine learning models to improve fraud detection accuracy. The goal is to reduce financial losses and improve customer trust for Adey Innovations Inc. through the deployment of robust fraud detection systems.

# Data Preparation
Feature and Target Separation: Features and target variables were separated for each dataset.
Train-Test Split: Both e-commerce and credit card datasets were split into training and testing sets.
Model Selection and Training
Models Used: Logistic regression, decision trees, random forest, gradient boosting, MLP, RNN, and LSTM.
Training Process: Models were trained on both datasets, and performance was evaluated using accuracy, precision, recall, and F1-score.
Evaluation Metrics: Each model was assessed, revealing that random forest and gradient boosting performed well on both datasets.
MLOps Integration
MLflow: Versioning and experiment tracking were enabled using MLflow to monitor model performance and parameters across different runs.
Model Evaluation
For Credit Card Data
Top Performers: Random forest, logistic regression, and gradient boosting had the highest accuracy and F1-scores, indicating strong predictive power.
For Fraud Data
Top Performers: Random forest and gradient boosting stood out with the highest performance metrics, particularly in handling class imbalance.
# Conclusion
The models developed and evaluated for fraud detection in both e-commerce and credit card transactions have shown promising results. Random forest and gradient boosting models performed best, offering significant potential for real-time fraud detection. Ongoing optimization and feature enhancement are recommended to further refine these models.

# Installation
To get started, follow these steps:

bash
Copy code
# Clone the repository
git clone https://github.com/HaYyu-Ra/ecommerce_fraud_detection_analysis.git
# Navigate to the project directory
cd ecommerce_fraud_detection_analysis
# Install dependencies
pip install -r requirements.txt
Usage
Run the main script as detailed in the repository to start the analysis and model training process.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For inquiries or suggestions, reach out to [hayyu.ragea@gmail.com].

GitHub Link: Fraud Detection Repository

This project supports Adey Innovations Inc. in improving fraud detection technologies, reducing financial risk, and enhancing customer trust.
