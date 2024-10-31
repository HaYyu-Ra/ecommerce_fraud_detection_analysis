import pandas as pd
from flask import Flask, jsonify
from dash import Dash, html, dcc, Input, Output
import plotly.express as px

# Initialize Flask app
app = Flask(__name__)

# Data paths
fraud_data_path = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\ecommerce_fraud_detection_system\Data\processed\fraud_cleaned.csv"
ip_country_data_path = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\ecommerce_fraud_detection_system\Data\IpAddress_to_Country.csv"
creditcard_data_path = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\ecommerce_fraud_detection_system\Data\processed\creditcard_cleaned.csv"

# Function to load data safely
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            print(f"No data: The file {file_path} is empty.")
            return None
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"No data: The file {file_path} is empty.")
        return None
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

# Load datasets
fraud_data = load_data(fraud_data_path)
ip_country_data = load_data(ip_country_data_path)
creditcard_data = load_data(creditcard_data_path)

# Verify columns and adjust names as needed
if fraud_data is not None:
    print("Fraud Data Columns:", fraud_data.columns)

    # Rename 'purchase_time' to 'transaction_date' if needed
    if 'purchase_time' in fraud_data.columns:
        fraud_data.rename(columns={'purchase_time': 'transaction_date'}, inplace=True)

    # Create 'date' column from 'signup_time' if not already present
    if 'signup_time' in fraud_data.columns:
        fraud_data['date'] = pd.to_datetime(fraud_data['signup_time']).dt.date
    else:
        print("Column 'signup_time' not found in fraud_data. Cannot create 'date' column.")

    # Check for 'transaction_date' and convert it if it exists
    if 'transaction_date' in fraud_data.columns:
        fraud_data['transaction_date'] = pd.to_datetime(fraud_data['transaction_date']).dt.to_period('M')
    else:
        print("Column 'transaction_date' not found in fraud_data.")

# Load IP address to country mapping
if ip_country_data is not None:
    print("IP Country Data Columns:", ip_country_data.columns)

# Function to map IP addresses to countries
def map_ip_to_country(ip_address):
    if pd.isnull(ip_address) or not isinstance(ip_address, str):
        return 'Unknown'

    try:
        ip_num = int(ip_address.split('.')[-1])  # Convert IP to an integer for range checking
        country_row = ip_country_data[
            (ip_country_data['lower_bound_ip_address'] <= ip_num) & 
            (ip_country_data['upper_bound_ip_address'] >= ip_num)
        ]
        if not country_row.empty:
            return country_row['country'].values[0]
    except (ValueError, IndexError):
        return 'Unknown'
    
    return 'Unknown'

# Add country information to fraud data
if fraud_data is not None and 'ip_address' in fraud_data.columns:
    fraud_data['country'] = fraud_data['ip_address'].apply(map_ip_to_country)
else:
    print("Column 'ip_address' not found in fraud_data.")

# Create a root route
@app.route('/')
def index():
    return "Welcome to the Fraud Detection API. Go to <a href='/dash/'>/dash/</a> for the dashboard."

# Flask endpoint for fraud statistics
@app.route('/api/fraud_stats', methods=['GET'])
def fraud_stats():
    total_transactions = len(fraud_data) if fraud_data is not None else 0
    total_frauds = fraud_data['class'].sum() if fraud_data is not None else 0
    fraud_percentage = (total_frauds / total_transactions * 100) if total_transactions > 0 else 0

    stats = {
        'total_transactions': total_transactions,
        'total_frauds': total_frauds,
        'fraud_percentage': fraud_percentage,
    }
    return jsonify(stats)

# Dash app for visualization
dash_app = Dash(__name__, server=app, routes_pathname_prefix='/dash/')

# Layout for the Dash app
dash_app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    html.Div(id='summary', style={'display': 'flex', 'justify-content': 'space-around', 'margin': '20px'}),
    dcc.Graph(id='fraud-trend-graph'),
    dcc.Graph(id='fraud-by-country'),
    dcc.Graph(id='fraud-by-device'),
])

# Update summary boxes
@dash_app.callback(
    Output('summary', 'children'),
    Input('fraud-trend-graph', 'id')
)
def update_summary_boxes(_):
    total_transactions = len(fraud_data) if fraud_data is not None else 0
    total_frauds = fraud_data['class'].sum() if fraud_data is not None else 0
    fraud_percentage = (total_frauds / total_transactions * 100) if total_transactions > 0 else 0

    return [
        html.Div(f"Total Transactions: {total_transactions}", style={'border': '1px solid black', 'padding': '10px', 'flex': '1'}),
        html.Div(f"Total Frauds: {total_frauds}", style={'border': '1px solid black', 'padding': '10px', 'flex': '1'}),
        html.Div(f"Fraud Percentage: {fraud_percentage:.2f}%", style={'border': '1px solid black', 'padding': '10px', 'flex': '1'}),
    ]

# Fraud trend graph by month
@dash_app.callback(
    Output('fraud-trend-graph', 'figure'),
    Input('fraud-trend-graph', 'id')
)
def update_fraud_trend_graph(_):
    if fraud_data is not None and 'date' in fraud_data.columns:
        try:
            fraud_trend = fraud_data.groupby('date')['class'].sum().reset_index()
            fraud_trend['date'] = pd.to_datetime(fraud_trend['date'])
            fig = px.line(fraud_trend, x='date', y='class', title='Monthly Fraud Cases Over Time')
            return fig
        except KeyError as e:
            print(f"Key error: {e}. Ensure the necessary columns exist.")
            return {}
    else:
        print("No data available for fraud trend graph.")
        return {}

# Fraud by country world map (all countries)
@dash_app.callback(
    Output('fraud-by-country', 'figure'),
    Input('fraud-by-country', 'id')
)
def update_fraud_by_country_graph(_):
    if fraud_data is not None and 'country' in fraud_data.columns:
        # Count fraud cases per country
        fraud_by_country = fraud_data['country'].value_counts().reset_index()  
        fraud_by_country.columns = ['country', 'number_of_frauds']

        # Create a world map
        fig = px.choropleth(
            fraud_by_country,
            locations='country',
            locationmode='country names',
            color='number_of_frauds',
            color_continuous_scale='YlOrBr',  # Yellow color scale
            title='Fraud Cases by Country',
            labels={'number_of_frauds': 'Number of Frauds'}
        )
        
        # Update the color to yellow
        fig.update_traces(marker=dict(line=dict(width=0.5, color='black')))  # Optional: add black border to countries
        fig.update_geos(showcoastlines=True, coastlinecolor="Black",  # Show coastlines
                        landcolor="white", bgcolor='lightgrey')  # Background color
        return fig
    else:
        print("No data available for fraud by country graph.")
        return {}

# Fraud by device ID bar chart
@dash_app.callback(
    Output('fraud-by-device', 'figure'),
    Input('fraud-by-device', 'id')
)
def update_fraud_by_device_graph(_):
    if fraud_data is not None and 'device_id' in fraud_data.columns:  # Changed from 'device_type' to 'device_id'
        fraud_by_device = fraud_data['device_id'].value_counts().reset_index()  # Change to use device IDs
        fraud_by_device.columns = ['device_id', 'number_of_frauds']
        fig = px.bar(fraud_by_device, x='device_id', y='number_of_frauds', title='Fraud Cases by Device ID')
        return fig
    else:
        print("No data available for fraud by device graph.")
        return {}

if __name__ == '__main__':
    app.run(debug=True)
