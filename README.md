# Fraud Detection Project

## Overview
This project focuses on enhancing fraud detection systems for Adey Innovations Inc. through advanced machine learning techniques, data preprocessing, and visualization.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Task 1: Data Preparation](#task-1-data-preparation)
4. [Task 2: Model Training](#task-2-model-training)
5. [Task 3: Model Explainability Report](#task-3-model-explainability-report)
6. [Task 4: Model Deployment and API Development](#task-4-model-deployment-and-api-development)
7. [Task 5: Build a Dashboard with Flask and Dash](#task-5-build-a-dashboard-with-flask-and-dash)
8. [Critical Thinking](#critical-thinking)
9. [Challenge](#challenge)
10. [General Conclusion](#general-conclusion)
11. [GitHub Link](#github-link)

## Installation
To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HaYyu-Ra/ecommerce_fraud_detection_analysis.git
   cd ecommerce_fraud_detection_analysis

    Set up a virtual environment (optional but recommended):

    bash

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:

bash

    pip install -r requirements.txt

Usage

To use the fraud detection system, follow these steps:

    Run the Flask API:

    bash

    python serve_model.py

    This will start the Flask application on http://127.0.0.1:5000.

    Access the API endpoints:
        GET /: Health check endpoint to ensure the API is running.
        POST /predict: Send a JSON payload to this endpoint to get predictions on potential fraud.

    Access the Dashboard: Open your web browser and go to http://127.0.0.1:8050 to view the interactive dashboard created with Dash.

Task 1: Data Preparation

The initial dataset was processed and cleaned to remove missing values, drop unnecessary features, and convert datetime columns into numeric features. The target variable was defined, and data was split into training and testing sets.

Key Steps:

    Checked for NaN values and dropped columns with more than 50% missing data.
    Converted datetime features to Unix timestamps.
    Used target encoding for high-cardinality categorical features.
    Defined features and target variable, ensuring the feature set was not empty.

Task 2: Model Training

A Random Forest classifier was trained on the processed dataset to predict fraudulent transactions. This model was selected due to its robustness and effectiveness in classification tasks.
Task 3: Model Explainability Report

Objective: The primary objective of this task was to enhance the explainability of the machine-learning model built for fraud detection by utilizing SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations).
1. Data Preparation:

The dataset was processed and cleaned, and the target variable was defined.
2. Model Training:

A Random Forest classifier was trained to predict fraudulent transactions.
3. Using SHAP for Explainability:

SHAP values were calculated to provide insights into the model's feature importance.

Insights from SHAP:

    Summary Plot: Highlights influential features affecting predictions.
    Force Plot: Visualizes feature contributions for specific predictions.
    Dependence Plot: Shows the relationship between "purchase_value" and prediction probability.

4. Using LIME for Explainability:

LIME was utilized to explain individual predictions, providing local interpretability.

Insights from LIME:

    Feature Importance Plot: Highlights which features contributed most significantly to specific predictions.

Key Insights and Conclusions:

    Model Trust and Transparency: Increased understanding of predictions.
    Feature Importance Identification: Features like "purchase_value" and "source" were significant.
    Improvement Opportunities: Identified areas for further investigation.

Task 4: Model Deployment and API Development
Results

    Setting Up the Flask API
        Directory Creation: Created a directory named fraud_detection_api.
        Flask Application Creation: Developed serve_model.py for the Flask API.
        Requirements File: Listed necessary dependencies in requirements.txt.

    API Development
        API Endpoints:
            GET /: Health check endpoint.
            POST /predict: Prediction endpoint for incoming data.
        Testing the API: Verified functionality using Postman.

    Dockerizing the Flask Application
        Dockerfile Creation: Defined the environment for deploying the Flask application.
        Building the Docker Image: Built using docker build -t fraud-detection-model ..
        Running the Docker Container: Launched with docker run -p 5000:5000 fraud-detection-model.

    Integration of Flask Logging
        Logging Setup: Configured logging for monitoring and debugging.

Task 5: Build a Dashboard with Flask and Dash - Result Report
Overview

Built an interactive dashboard using Flask as the backend and Dash for visualizations.
Key Features Implemented

    Flask Endpoint for Fraud Statistics: Created an API endpoint for summary statistics.
    Dash Frontend: Created interactive visualizations.

Output Results

    Loading and Processing Data: Successfully loaded data.
    API Endpoint for Fraud Statistics: Returns a JSON object with summary statistics.
    Dashboard Insights: Provides valuable insights into fraud detection.

Critical Thinking

In tackling the challenge of improving fraud detection, several critical considerations emerged.
Challenge

Several challenges were encountered throughout the project, particularly with handling imbalanced datasets and technical aspects of model deployment.
General Conclusion

The project has successfully demonstrated the potential of advanced machine learning techniques combined with robust data pre-processing and feature engineering.
GitHub Link

Clone the repository to access the project:

bash

git clone https://github.com/HaYyu-Ra/ecommerce_fraud_detection_analysis.git
