import pandas as pd
from sklearn.model_selection import train_test_split

# Define file paths
fraud_data_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\fraud_detection_project\data\processed\fraud_cleaned.csv'
credit_card_data_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\fraud_detection_project\data\processed\creditcard_cleaned.csv'

# Load the datasets
fraud_data = pd.read_csv(fraud_data_path)
credit_card_data = pd.read_csv(credit_card_data_path)

# Data Preparation for Fraud_Data (E-commerce fraud dataset)

# Separate features (X) and target (y)
X_fraud = fraud_data.drop(columns=['class'])  # Dropping the target column 'class'
y_fraud = fraud_data['class']  # Target column

# Train-test split for Fraud_Data
X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(
    X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud)

print(f"Fraud_Data: Training set size: {X_fraud_train.shape}, Test set size: {X_fraud_test.shape}")

# Data Preparation for CreditCard Data (Bank transaction fraud dataset)

# Separate features (X) and target (y)
X_creditcard = credit_card_data.drop(columns=['Class'])  # Dropping the target column 'Class'
y_creditcard = credit_card_data['Class']  # Target column

# Train-test split for CreditCard Data
X_creditcard_train, X_creditcard_test, y_creditcard_train, y_creditcard_test = train_test_split(
    X_creditcard, y_creditcard, test_size=0.2, random_state=42, stratify=y_creditcard)

print(f"CreditCard_Data: Training set size: {X_creditcard_train.shape}, Test set size: {X_creditcard_test.shape}")

# Save the train-test split data for future use (optional)
X_fraud_train.to_csv('fraud_train_features.csv', index=False)
X_fraud_test.to_csv('fraud_test_features.csv', index=False)
y_fraud_train.to_csv('fraud_train_labels.csv', index=False)
y_fraud_test.to_csv('fraud_test_labels.csv', index=False)

X_creditcard_train.to_csv('creditcard_train_features.csv', index=False)
X_creditcard_test.to_csv('creditcard_test_features.csv', index=False)
y_creditcard_train.to_csv('creditcard_train_labels.csv', index=False)
y_creditcard_test.to_csv('creditcard_test_labels.csv', index=False)
