# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set(style="whitegrid")

# Paths to cleaned data
fraud_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/fraud_cleaned.csv"
creditcard_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/creditcard_cleaned.csv"
ip_country_data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/fraud_detection_project/data/processed/ip_country_cleaned.csv"

# Load datasets
fraud_df = pd.read_csv(fraud_data_path)
creditcard_df = pd.read_csv(creditcard_data_path)
ip_country_df = pd.read_csv(ip_country_data_path)

# Function for univariate analysis
def univariate_analysis():
    # Fraud class distribution
    plt.figure(figsize=(12, 6))

    # E-commerce Fraud
    plt.subplot(1, 2, 1)
    sns.countplot(data=fraud_df, x='class')
    plt.title('E-commerce Fraud Class Distribution')
    plt.xlabel('Fraud Class')
    plt.ylabel('Count')

    # Credit Card Fraud
    plt.subplot(1, 2, 2)
    sns.countplot(data=creditcard_df, x='Class')
    plt.title('Credit Card Fraud Class Distribution')
    plt.xlabel('Fraud Class')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

# Function for bivariate analysis
def bivariate_analysis():
    # Correlation Heatmap for credit card dataset
    plt.figure(figsize=(12, 8))
    corr_matrix = creditcard_df.corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
    plt.title('Correlation Heatmap for Credit Card Transactions')
    plt.show()

    # Device ID and fraud in e-commerce
    plt.figure(figsize=(10, 6))
    sns.countplot(data=fraud_df, x='device_id', hue='class', order=fraud_df['device_id'].value_counts().index[:10])
    plt.title('Top 10 Devices by Fraud Class (E-commerce)')
    plt.xlabel('Device ID')
    plt.ylabel('Count')
    plt.legend(title='Fraud Class')
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Performing Univariate Analysis...")
    univariate_analysis()

    print("Performing Bivariate Analysis...")
    bivariate_analysis()

    print("EDA completed.")
