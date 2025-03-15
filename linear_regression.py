from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

def liner_regressor(x_train, y_train):
    # some code here
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train) # Here it will be use function cost to do optimization model
    
    preds = lin_reg.predict(x_train)
    lin_mse = mean_squared_error(y_train, preds)
    
    lin_rmse = np.sqrt(lin_mse)
    
    print(lin_rmse)
    
    lin_scores = cross_val_score(lin_reg, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    
    print('Lin_Scores:', lin_rmse_scores)
    print('Mean:', lin_rmse_scores.mean())
    print('Standard Deviation:', lin_rmse_scores.std())