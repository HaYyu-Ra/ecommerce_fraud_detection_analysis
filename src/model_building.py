# Import statements
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dataset paths
credit_card_file_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\ecommerce_fraud_detection_system\Data\processed\creditcard_cleaned.csv'
fraud_data_file_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\ecommerce_fraud_detection_system\Data\processed\merged_data.csv'

# Load datasets with error handling
def load_datasets():
    try:
        fraud_data = pd.read_csv(fraud_data_file_path)
        creditcard_data = pd.read_csv(credit_card_file_path)
        logging.info("Datasets loaded successfully.")
        return fraud_data, creditcard_data
    except FileNotFoundError as e:
        logging.error(f"Error loading dataset: {e}")
        raise

fraud_data, creditcard_data = load_datasets()

# Data Preparation: Feature and Target Separation
def prepare_data(fraud_data, creditcard_data):
    X_fraud = fraud_data.drop('class', axis=1)
    y_fraud = fraud_data['class']
    X_credit = creditcard_data.drop('Class', axis=1)
    y_credit = creditcard_data['Class']
    return X_fraud, y_fraud, X_credit, y_credit

X_fraud, y_fraud, X_credit, y_credit = prepare_data(fraud_data, creditcard_data)

# Preprocessing Function
def preprocess_data(X):
    # Convert datetime columns to int64 and drop any columns that cannot be converted
    for col in X.select_dtypes(include=['object']):
        try:
            X[col] = pd.to_datetime(X[col], format='%Y-%m-%d', errors='coerce').astype(np.int64) // 10**9
        except ValueError:
            X.drop(col, axis=1, inplace=True)
    
    # Drop columns that are entirely NaN
    X.dropna(axis=1, how='all', inplace=True)

    # Drop columns with a high percentage of missing values
    threshold = 0.5
    X = X.loc[:, X.isnull().mean() < threshold]

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    logging.info("Data preprocessing completed.")
    return pd.DataFrame(X_imputed, columns=X.columns)

X_fraud = preprocess_data(X_fraud)
X_credit = preprocess_data(X_credit)

# Train-Test Split
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = split_data(X_fraud, y_fraud)
X_train_credit, X_test_credit, y_train_credit, y_test_credit = split_data(X_credit, y_credit)

# Standard Scaling
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

X_train_fraud_scaled, X_test_fraud_scaled = scale_data(X_train_fraud, X_test_fraud)
X_train_credit_scaled, X_test_credit_scaled = scale_data(X_train_credit, X_test_credit)

# Model Selection: Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP": MLPClassifier(max_iter=500),
}

# Initialize MLflow for Experiment Tracking
mlflow.set_experiment("Fraud Detection Experiment")

# Model Training and Evaluation Function
def train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, X_test_original):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        
        report = classification_report(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        mlflow.log_param("model", model_name)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_text(report, "classification_report.txt")
        
        # Log model with an input example if available
        if not X_test_original.empty:
            try:
                input_example = X_test_original.iloc[:1]
                mlflow.sklearn.log_model(model, model_name, input_example=input_example)
            except Exception as e:
                logging.error(f"Error logging model: {e}")
        
        logging.info(f"Model: {model_name}\n{report}\nAUC: {auc}")

# Training traditional ML models on Fraud Data
for model_name, model in models.items():
    train_and_evaluate_model(model_name, model, X_train_fraud_scaled, y_train_fraud, X_test_fraud_scaled, y_test_fraud, X_test_fraud)

# Training traditional ML models on Credit Card Data
for model_name, model in models.items():
    train_and_evaluate_model(model_name, model, X_train_credit_scaled, y_train_credit, X_test_credit_scaled, y_test_credit, X_test_credit)

# Deep Learning Models: CNN, RNN, and LSTM
def build_and_train_nn_model(model_type, X_train, y_train, X_test, y_test, input_shape):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    if model_type == "CNN":
        x = tf.keras.layers.Conv1D(64, 3, activation="relu")(input_layer)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Flatten()(x)
    elif model_type == "RNN":
        x = tf.keras.layers.SimpleRNN(64)(input_layer)
    elif model_type == "LSTM":
        x = tf.keras.layers.LSTM(64)(input_layer)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    with mlflow.start_run(run_name=f"{model_type} Model"):
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
        
        # Log model with an input example if available
        if len(X_test) > 0:
            try:
                input_example = X_test[:1]
                signature = mlflow.models.infer_signature(X_train, model.predict(X_train[:1]))
                mlflow.keras.log_model(model, model_type, input_example=input_example, signature=signature)
                mlflow.log_artifacts("classification_report.txt")
            except Exception as e:
                logging.error(f"Error logging model: {e}")

        for epoch, acc in enumerate(history.history['accuracy']):
            mlflow.log_metric("train_accuracy", acc, step=epoch)
        for epoch, val_acc in enumerate(history.history['val_accuracy']):
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

    return model

# Reshape data for deep learning models
X_train_fraud_reshaped = X_train_fraud_scaled.reshape(-1, X_train_fraud_scaled.shape[1], 1)
X_test_fraud_reshaped = X_test_fraud_scaled.reshape(-1, X_test_fraud_scaled.shape[1], 1)

X_train_credit_reshaped = X_train_credit_scaled.reshape(-1, X_train_credit_scaled.shape[1], 1)
X_test_credit_reshaped = X_test_credit_scaled.reshape(-1, X_test_credit_scaled.shape[1], 1)

# Build and Train Deep Learning Models
cnn_model = build_and_train_nn_model("CNN", X_train_fraud_reshaped, y_train_fraud, X_test_fraud_reshaped, y_test_fraud, (X_train_fraud_reshaped.shape[1], 1))
rnn_model = build_and_train_nn_model("RNN", X_train_fraud_reshaped, y_train_fraud, X_test_fraud_reshaped, y_test_fraud, (X_train_fraud_reshaped.shape[1], 1))
lstm_model = build_and_train_nn_model("LSTM", X_train_fraud_reshaped, y_train_fraud, X_test_fraud_reshaped, y_test_fraud, (X_train_fraud_reshaped.shape[1], 1))

# Further evaluation and analysis can follow here
