import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def tree_regressor(x_train, y_train):
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(x_train, y_train)
    
    
    preds_tree = tree_reg.predict(x_train)
    tree_mse = mean_squared_error(y_train, preds_tree)
    
    tree_rmse = np.sqrt(tree_mse)
    
    print(tree_rmse)
    
    scores = cross_val_score(tree_reg, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    
    print('Tre_Scores:', tree_rmse_scores)
    print('Mean:', tree_rmse_scores.mean())
    print('Standard Deviation:', tree_rmse_scores.std())