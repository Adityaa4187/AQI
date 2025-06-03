from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, mean_squared_log_error
import numpy as np

def evaluate_model(rf_model, X_train_features, y_train, X_val_features, y_val, X_test_features, y_test, power_transformer):
    """
    Evaluates the Random Forest model performance on training, validation, and test sets.
    :param rf_model: Trained Random Forest model.
    :param X_train_features: Features for the training set.
    :param y_train: True labels for the training set.
    :param X_val_features: Features for the validation set.
    :param y_val: True labels for the validation set.
    :param X_test_features: Features for the test set.
    :param y_test: True labels for the test set.
    :param power_transformer: PowerTransformer used for scaling target values.
    :return: None
    """
    # Predictions on training set
    y_train_pred = rf_model.predict(X_train_features)
    y_train_pred_inv = power_transformer.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
    y_train_inv = power_transformer.inverse_transform(y_train.reshape(-1, 1)).flatten()

    # Evaluation on training set
    print("Training Set Metrics:")
    print(f'Training MAE: {mean_absolute_error(y_train_inv, y_train_pred_inv)}')
    print(f'Training MSE: {mean_squared_error(y_train_inv, y_train_pred_inv)}')
    print(f'Training RMSE: {np.sqrt(mean_squared_error(y_train_inv, y_train_pred_inv))}')
    print(f'Training R-squared: {r2_score(y_train_inv, y_train_pred_inv)}')
    print(f'Training Explained Variance Score: {explained_variance_score(y_train_inv, y_train_pred_inv)}')
    print(f'Training Mean Squared Logarithmic Error: {mean_squared_log_error(y_train_inv, y_train_pred_inv)}')

    # Predictions on validation set
    y_val_pred = rf_model.predict(X_val_features)
    y_val_pred_inv = power_transformer.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
    y_val_inv = power_transformer.inverse_transform(y_val.reshape(-1, 1)).flatten()

    # Evaluation on validation set
    print("\nValidation Set Metrics:")
    print(f'Validation MAE: {mean_absolute_error(y_val_inv, y_val_pred_inv)}')
    print(f'Validation MSE: {mean_squared_error(y_val_inv, y_val_pred_inv)}')
    print(f'Validation RMSE: {np.sqrt(mean_squared_error(y_val_inv, y_val_pred_inv))}')
    print(f'Validation R-squared: {r2_score(y_val_inv, y_val_pred_inv)}')
    print(f'Validation Explained Variance Score: {explained_variance_score(y_val_inv, y_val_pred_inv)}')
    print(f'Validation Mean Squared Logarithmic Error: {mean_squared_log_error(y_val_inv, y_val_pred_inv)}')

    # Predictions on test set
    y_test_pred = rf_model.predict(X_test_features)
    y_test_pred_inv = power_transformer.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
    y_test_inv = power_transformer.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Evaluation on test set
    print("\nTest Set Metrics:")
    print(f'Test MAE: {mean_absolute_error(y_test_inv, y_test_pred_inv)}')
    print(f'Test MSE: {mean_squared_error(y_test_inv, y_test_pred_inv)}')
    print(f'Test RMSE: {np.sqrt(mean_squared_error(y_test_inv, y_test_pred_inv))}')
    print(f'Test R-squared: {r2_score(y_test_inv, y_test_pred_inv)}')
    print(f'Test Explained Variance Score: {explained_variance_score(y_test_inv, y_test_pred_inv)}')
    print(f'Test Mean Squared Logarithmic Error: {mean_squared_log_error(y_test_inv, y_test_pred_inv)}')
