import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Dict, Any

# Load your data
def load_data(file_paths: Tuple[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fraud_data = pd.read_csv(file_paths[0])
    credit_data = pd.read_csv(file_paths[1])
    return fraud_data, credit_data

# Preprocess data by converting non-numeric types and handling missing values
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Convert date columns to numeric (timestamp)
    for col in data.select_dtypes(include=['object']):
        try:
            data[col] = pd.to_datetime(data[col]).astype(int) // 10**9  # Convert to UNIX timestamp
        except (ValueError, TypeError):
            data.drop(columns=[col], inplace=True)  # Drop non-convertible columns

    # Handle missing values (if any)
    data.fillna(0, inplace=True)  # Or use other imputation methods

    return data

# Prepare data for modeling
def prepare_data(data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = preprocess_data(data)  # Preprocess the data here
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate a model
def evaluate_model(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, model_name: str) -> None:
    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Convert probabilities to binary if necessary
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = (predictions > 0.5).astype(int)
        elif predictions.ndim == 1 and not np.issubdtype(predictions.dtype, np.integer):
            predictions = (predictions > 0.5).astype(int)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)

        # Log parameters, metrics, and model
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("num_features", X_train.shape[1])  # Number of features
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(report, "classification_report.txt")
        
        # Log models depending on type
        if isinstance(model, keras.Model):
            mlflow.tensorflow.log_model(model, model_name)
        else:
            mlflow.sklearn.log_model(model, model_name)

        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy}")
        print(report)

# Build CNN model
def build_cnn_model(input_shape: Tuple[int, int, int]) -> keras.Model:
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build RNN model
def build_rnn_model(input_shape: Tuple[int, int]) -> keras.Model:
    model = keras.Sequential([
        layers.SimpleRNN(50, input_shape=input_shape, return_sequences=True),
        layers.SimpleRNN(50),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build LSTM model
def build_lstm_model(input_shape: Tuple[int, int]) -> keras.Model:
    model = keras.Sequential([
        layers.LSTM(50, input_shape=input_shape, return_sequences=True),
        layers.LSTM(50),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Automatically adjust reshaping
def auto_reshape(X: pd.DataFrame, model_type: str) -> Tuple:
    num_samples, num_features = X.shape
    
    if model_type == 'cnn':
        side_length = int(num_features ** 0.5)
        if side_length ** 2 != num_features:
            raise ValueError(f"Cannot reshape array of size {X.size} into a square for CNN.")
        return X.values.reshape(num_samples, side_length, side_length, 1)
    
    elif model_type in ['rnn', 'lstm']:
        return X.values.reshape(num_samples, num_features, 1)
    
    else:
        raise ValueError(f"Unsupported model type for reshaping: {model_type}")

# Evaluate models
def evaluate_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "MLP Classifier": MLPClassifier(max_iter=1000),
    }

    # Evaluate classical machine learning models
    for model_name, model in models.items():
        evaluate_model(model, X_train, X_test, y_train, y_test, model_name)

    # Reshape data for CNN, RNN, LSTM
    try:
        X_train_cnn = auto_reshape(X_train, 'cnn')
        X_test_cnn = auto_reshape(X_test, 'cnn')
        evaluate_model(build_cnn_model(X_train_cnn.shape[1:]), X_train_cnn, X_test_cnn, y_train, y_test, "CNN Model")
    except ValueError as e:
        print(f"Skipping CNN evaluation: {e}")

    # Reshape for RNN and LSTM
    try:
        X_train_rnn = auto_reshape(X_train, 'rnn')
        X_test_rnn = auto_reshape(X_test, 'rnn')
        
        evaluate_model(build_rnn_model(X_train_rnn.shape[1:]), X_train_rnn, X_test_rnn, y_train, y_test, "RNN Model")
        evaluate_model(build_lstm_model(X_train_rnn.shape[1:]), X_train_rnn, X_test_rnn, y_train, y_test, "LSTM Model")
    except ValueError as e:
        print(f"Skipping RNN/LSTM evaluation: {e}")

if __name__ == "__main__":
    # Set experiment name and create it if it doesn't exist
    experiment_name = "fraud_detection_experiment"
    
    # Set the tracking URI for MLflow
    mlflow.set_tracking_uri("file:./mlruns")  # Ensure this path is writable
    
    # Create the experiment if it does not exist
    try:
        mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException as e:
        if "already exists" in str(e):
            print(f"Experiment '{experiment_name}' already exists. Using the existing experiment.")
    
    # Set the current experiment
    mlflow.set_experiment(experiment_name)

    # Load data
    file_paths = (
        r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\ecommerce_fraud_detection_system\Data\processed\fraud_cleaned.csv',
        r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\ecommerce_fraud_detection_system\Data\processed\creditcard_cleaned.csv'
    )
    fraud_data, credit_data = load_data(file_paths)

    # Prepare and evaluate models for credit card data
    X_train_cc, X_test_cc, y_train_cc, y_test_cc = prepare_data(credit_data, 'Class')
    print("Evaluating models for Credit Card Data:")
    evaluate_models(X_train_cc, X_test_cc, y_train_cc, y_test_cc)

    # Prepare and evaluate models for fraud data
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = prepare_data(fraud_data, 'class')  # Ensure 'class' is the correct target column
    print("Evaluating models for Fraud Data:")
    evaluate_models(X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud)
