# src/data_preprocessing.py

import pandas as pd

def load_data():
    fraud_data = pd.read_csv(r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\ecommerce_fraud_detection_system\Data\processed\fraud_with_country.csv')
    credit_data = pd.read_csv(r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\ecommerce_fraud_detection_system\Data\processed\creditcard_cleaned.csv')
    return fraud_data, credit_data

def preprocess_fraud_data(fraud_data):
    # Convert timestamps to datetime
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    
    # Drop unnecessary columns
    fraud_data = fraud_data.drop(columns=['device_id'])
    
    return fraud_data

def preprocess_credit_data(credit_data):
    # Normalize Amount
    credit_data['Amount'] = (credit_data['Amount'] - credit_data['Amount'].mean()) / credit_data['Amount'].std()
    return credit_data

if __name__ == '__main__':
    fraud_data, credit_data = load_data()
    fraud_data = preprocess_fraud_data(fraud_data)
    credit_data = preprocess_credit_data(credit_data)
