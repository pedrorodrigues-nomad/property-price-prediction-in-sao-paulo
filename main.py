import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from linear_regression import liner_regressor
from tree_regressor import tree_regressor
from forest_regressor import forest_regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

# Set Mapbox access token
px.set_mapbox_access_token(open("mapbox_token").read())

# Load data
try:
    df_data = pd.read_csv("sao-paulo-properties-april-2019.csv")
except pd.errors.ParserError as e:
    print(f"Error reading CSV file: {e}")
    df_data = pd.DataFrame()  # Create an empty DataFrame as a fallback

def main():
    X, Y = data()
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    # Test different models
    liner_regressor(x_train, y_train)
    tree_regressor(x_train, y_train)
    forest_regressor(x_train, y_train)

    # Implementing the best model: RandomForest Regressor
    params_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    
    forest_reg = RandomForestRegressor()
    
    try:
        grid_search = GridSearchCV(forest_reg, params_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(x_train, y_train)
    except Exception as e:
        print(f"Error during grid search: {e}")
        return
    
    final_model = grid_search.best_estimator_
    final_model_predictions = final_model.predict(x_test)
    
    final_model_mse = mean_squared_error(y_test, final_model_predictions)
    final_model_rmse = np.sqrt(final_model_mse)
    
    # Save cross-validation results
    exportcv = pd.DataFrame(grid_search.cv_results_)
    exportcv.to_csv('cv_results.csv')
    
    print(f"Final Model RMSE: {final_model_rmse}")
    
    # Plot results
    fig = go.Figure(data=[
        go.Scatter(y=y_test.values, mode='lines', name='Actual'),
        go.Scatter(y=final_model_predictions, mode='lines', name='Predicted')
    ])
    fig.update_layout(title='Actual vs Predicted Prices', xaxis_title='Index', yaxis_title='Price')
    fig.show()
    
    print("Done")
    
    # Save the model
    save_model(final_model)

def data():
    df_rent = df_data[df_data["Negotiation Type"] == "rent"]
    df_cleaned = df_rent.drop(['New', 'Property Type', 'Negotiation Type'], axis=1)
    one_hot = pd.get_dummies(df_cleaned['District'])
    df = df_cleaned.drop('District', axis=1)
    df = df.join(one_hot)
    
    Y = df['Price']
    X = df.loc[:, df.columns != 'Price']
    
    return X, Y

def save_model(model):
    joblib.dump(model, 'final_model.joblib')
    print("Model saved as final_model.joblib")

if __name__ == "__main__":
    main()