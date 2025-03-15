import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def forest_regressor(x_train, y_train):
    
    forest_reg = RandomForestRegressor()
    forest_reg.fit(x_train, y_train)
    
    preds_forest = forest_reg.predict(x_train)
    forest_mse = mean_squared_error(y_train, preds_forest)
    
    forest_rmse = np.sqrt(forest_mse)
    
    print(forest_rmse)
    
    forest_scores = cross_val_score(forest_reg, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    
    print('Forest_Scores:', forest_rmse_scores)
    print('Mean:', forest_rmse_scores.mean())
    print('Standard Deviation:', forest_rmse_scores.std())