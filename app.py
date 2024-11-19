import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load datasets
file_paths = ['2010_b.csv', '2015_b.csv', '2020_b.csv']
datasets = [pd.read_csv(file) for file in file_paths]
data = pd.concat(datasets, ignore_index=True)

# Filter numeric columns for visualization
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Prepare data for machine learning
data_ml = data[['DATE', 'AIR_TEMP', 'SEA_SURF_TEMP', 'WIND_SPEED', 'SEA_LVL_PRES']].dropna()
data_ml['DATE'] = pd.to_datetime(data_ml['DATE'])
data_ml['YEAR'] = data_ml['DATE'].dt.year
data_ml['MONTH'] = data_ml['DATE'].dt.month

# Features and targets for prediction
features = ['YEAR', 'MONTH', 'AIR_TEMP', 'WIND_SPEED', 'SEA_LVL_PRES']
X = data_ml[features]  # Define X here, this is the feature set
targets = ['SEA_SURF_TEMP', 'AIR_TEMP', 'WIND_SPEED', 'SEA_LVL_PRES']

# Train RandomForest models for all targets
models = {}
for target in targets:
    y = data_ml[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    models[target] = model

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("BlueSegara App"),
    
    dcc.Dropdown(
        id='parameter-dropdown',
        options=[{'label': col.replace('_', ' ').title(), 'value': col} for col in numeric_columns],
        value='SEA_SURF_TEMP',  # Default parameter to 'SEA_SURF_TEMP'
        style={'width': '50%'}
    ),
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': str(year), 'value': year} for year in range(2010, 2041, 5)],  # 5-year intervals, starting from 2010
        value=2020,  # Default year to 2020
        style={'width': '50%'}
    ),
    dcc.Graph(id='map-plot'),
    dcc.Graph(id='prediction-plot')
])

# Callback to update the map plot
@app.callback(
    Output('map-plot', 'figure'),
    Input('parameter-dropdown', 'value')
)
def update_map(selected_param):
    fig = px.scatter_geo(
        data,
        lat='LATITUDE',
        lon='LONGITUDE',
        color=selected_param,
        title=f"Distribusi {selected_param.replace('_', ' ').title()}",
        projection="natural earth",
        hover_name=selected_param
    )
    return fig

# Callback to update prediction plot
@app.callback(
    Output('prediction-plot', 'figure'),
    Input('year-dropdown', 'value')
)
def update_predictions(selected_year):
    # Prepare future data for all parameters to be predicted
    future_data = pd.DataFrame({
        'YEAR': [selected_year] * 12,
        'MONTH': list(range(1, 13)),
        'AIR_TEMP': [data_ml['AIR_TEMP'].mean()] * 12,  # Use mean value for simplicity
        'WIND_SPEED': [data_ml['WIND_SPEED'].mean()] * 12,  # Use mean value for simplicity
        'SEA_LVL_PRES': [data_ml['SEA_LVL_PRES'].mean()] * 12  # Use mean value for simplicity
    })

    # Dictionary to store predictions for each target
    predictions = {}
    for target in targets:
        predictions[target] = models[target].predict(future_data)
    
    # Prepare data for visualization
    pred_df = pd.DataFrame(predictions)
    pred_df['MONTH'] = future_data['MONTH']
    
    # Plot predictions for all targets
    fig = px.line(
        pred_df, 
        x='MONTH', 
        y=targets, 
        title=f"Predictions for {selected_year}", 
        labels={'value': 'Predicted Value', 'variable': 'Parameter'}
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
