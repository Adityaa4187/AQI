from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model.
    :param X_train: Training feature data.
    :param y_train: Training target data.
    :return: Trained Random Forest model.
    """
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, max_features='sqrt', n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def tune_random_forest(X_train, y_train):
    """
    Perform hyperparameter tuning for the Random Forest model.
    :param X_train: Training feature data.
    :param y_train: Training target data.
    :return: Best Random Forest model after tuning.
    """
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, max_features='sqrt', n_jobs=-1, random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid, n_iter=10, n_jobs=-1, random_state=42)
    rf_random_search.fit(X_train, y_train)
    best_rf_model = rf_random_search.best_estimator_
    return best_rf_model
