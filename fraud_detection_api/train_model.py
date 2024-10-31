import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Set the data path
data_path = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\ecommerce_fraud_detection_system\fraud_detection_api\data\merged_data.csv"

# Load the dataset
df = pd.read_csv(data_path)

# Data Preprocessing
# Drop columns and rows with all missing values
df.dropna(axis=1, how='all', inplace=True)  # Drop empty columns
df.dropna(axis=0, how='any', inplace=True)  # Drop rows with any empty values

# Identify categorical and numerical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
numerical_columns = df.select_dtypes(exclude=['object']).columns.tolist()

# Define columns to handle based on cardinality
low_cardinality_columns = [col for col in categorical_columns if df[col].nunique() < 50]
high_cardinality_columns = [col for col in categorical_columns if df[col].nunique() >= 50]

# One-hot encode low-cardinality categorical columns
if low_cardinality_columns:
    df = pd.get_dummies(df, columns=low_cardinality_columns, drop_first=True)

# Label encode high-cardinality categorical columns
label_encoder = LabelEncoder()
for col in high_cardinality_columns:
    if col in df.columns:  # Check if column exists before encoding
        df[col] = label_encoder.fit_transform(df[col])

# Split features and target variable
if 'class' in df.columns:
    X = df.drop('class', axis=1)  # Features
    y = df['class']               # Target
else:
    raise ValueError("The target column 'class' is missing from the dataset.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Evaluate the model
y_pred = random_forest_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Random Forest Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Save the model to a file
model_path = os.path.join(os.path.dirname(data_path), 'random_forest_model.pkl')
joblib.dump(random_forest_model, model_path)
print(f"Random Forest Model saved to {model_path}")
