from data_preprocessing import load_data, remove_outliers_z_transform
from feature_scaling import scale_features
from cnn_lstm_model import build_cnn_lstm_model, train_cnn_lstm_model, extract_features
from random_forest_model import train_random_forest, tune_random_forest
from evaluation import evaluate_model
from visualization import plot_actual_vs_predicted, plot_learning_curves, plot_distribution_before_scaling, plot_boxplot_before_scaling, plot_distribution_after_scaling, plot_boxplot_after_scaling
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# Load and preprocess data
df = load_data(r"C:\Users\adity\OneDrive\Desktop\Peerj_changes\AQI_code\station_hour\station_hour.csv")
df = remove_outliers_z_transform(df, ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'])
scaled_df, power_transformer = scale_features(df, ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'], 'AQI')

# Visualization of distributions
plot_distribution_before_scaling(df, ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'], colors=['blue', 'green', 'red', 'purple', 'orange', 'brown'])
plot_distribution_after_scaling(scaled_df, ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'], colors=['blue', 'green', 'red', 'purple', 'orange', 'brown'])

# Visualization of boxplots
plot_boxplot_before_scaling(df, ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'], colors=['blue', 'green', 'red', 'purple', 'orange', 'brown'])
plot_boxplot_after_scaling(scaled_df, ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'], colors=['blue', 'green', 'red', 'purple', 'orange', 'brown'])

# Prepare input data
X = scaled_df[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']].values
y = scaled_df['AQI'].values

# Reshape the data for CNN-LSTM
timesteps = 10
X_reshaped = []
y_reshaped = []

for i in range(len(X) - timesteps):
    X_reshaped.append(X[i:i + timesteps])
    y_reshaped.append(y[i + timesteps])

X_reshaped = np.array(X_reshaped)
y_reshaped = np.array(y_reshaped)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build, train, and extract features using CNN-LSTM model
cnn_lstm_model = build_cnn_lstm_model(X_train.shape[1:])
history = train_cnn_lstm_model(cnn_lstm_model, X_train, y_train, X_val, y_val)
X_train_features = extract_features(cnn_lstm_model, X_train)
X_val_features = extract_features(cnn_lstm_model, X_val)
X_test_features = extract_features(cnn_lstm_model, X_test)

# Train and tune Random Forest model
rf_model = train_random_forest(X_train_features, y_train)
best_rf_model = tune_random_forest(X_train_features, y_train)

# Evaluate model
evaluate_model(best_rf_model, X_val_features, y_val, X_test_features, y_test, power_transformer)

# Visualize results
plot_learning_curves(history)
plot_actual_vs_predicted(y_test, best_rf_model.predict(X_test_features))
